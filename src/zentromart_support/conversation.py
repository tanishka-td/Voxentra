"""Conversation state, verification gate, deterministic handovers, and text replies."""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .agent import GroundedTextAgent
from .capstone_tools import CapstoneToolGateway


ORDER_ID_RE = re.compile(r"(?<!\d)(?:\d{8}|\d{4}[ -]\d{4})(?!\d)")
PROOF_RE = re.compile(r"(?<!\d)\d{4,6}(?!\d)")
ORDER_HELP_RE = re.compile(
    r"\b(my order|order about|order ke baare|help with (?:an? )?order|"
    r"order number|order id|track my|where is my order|my delivery|my refund|my shipment)\b|"
    r"ऑर्डर के बारे में|अपने ऑर्डर",
    re.IGNORECASE,
)
ORDER_FOLLOWUP_RE = re.compile(
    r"\b(it|that order|this order|the order|its status|order status|where is it|"
    r"when will it|has it|was it|did it|for that order)\b|"
    r"उसका स्टेटस|वह कब|कब आएगा|कहाँ है",
    re.IGNORECASE,
)
HUMAN_RE = re.compile(r"\b(human|person|agent|representative|speak to someone)\b|इंसान|व्यक्ति", re.I)
FRAUD_RE = re.compile(r"\b(never ordered|never placed|did not order|didn't order|unauthori[sz]ed|fraud|not mine)\b", re.I)
LEGAL_RE = re.compile(r"\b(legal action|consumer court|police complaint|police)\b", re.I)
SAFETY_RE = re.compile(r"\b(swelling battery|swollen battery|burning smell|sparks?|smoke|caught fire|injury|electric shock)\b", re.I)
DAMAGE_RE = re.compile(r"\b(damaged|damage|cracked|crack|broken|defective|wrong item|missing item|torn|expired)\b", re.I)

ORDER_FIELDS = (
    "order_id", "status", "placed_at", "payment_method", "delivery_type",
    "promised_date", "shipped_on", "delivered_on", "delivery_attempts",
    "subtotal_inr", "delivery_fee_inr", "cod_fee_inr", "total_inr",
    "coupon_code", "return_sku", "return_reason", "return_status",
    "return_requested_on", "return_pickup_on", "return_qc_passed_on",
    "return_refund_initiated_on", "return_refund_method",
    "return_refund_amount_inr", "return_refund_completed_on", "return_rrn",
    "cancellation_refund_initiated_on", "cancellation_refund_method",
    "cancellation_refund_amount_inr", "cancellation_rrn",
    "rto_refund_initiated_on", "rto_refund_method", "rto_refund_amount_inr",
    "rto_rrn", "express_late_credit_inr", "express_late_credit_on",
    "damage_sku", "damage_issue", "damage_reported_at", "damage_status",
    "timeline", "contacts_last_7_days",
)


def _ascii_digits(text: str) -> str:
    output = []
    for char in text:
        try:
            output.append(str(unicodedata.digit(char)))
        except (TypeError, ValueError):
            output.append(char)
    return "".join(output)


def _redact_proofs(text: str) -> str:
    normalized = _ascii_digits(text)
    return PROOF_RE.sub("[verification number]", normalized)


def _extract_order_id(text: str) -> str | None:
    normalized = _ascii_digits(text)
    match = ORDER_ID_RE.search(normalized)
    return re.sub(r"\D", "", match.group(0)) if match else None


def _extract_proof(text: str) -> str | None:
    normalized = _ascii_digits(text)
    order_match = ORDER_ID_RE.search(normalized)
    if order_match:
        normalized = normalized[:order_match.start()] + " " + normalized[order_match.end():]
    match = PROOF_RE.search(normalized)
    return match.group(0) if match else None


def _safe_order(order: dict[str, Any]) -> dict[str, Any]:
    """Keep only facts needed for support; omit customer contact and address data."""
    safe = {key: order[key] for key in ORDER_FIELDS if order.get(key) is not None}
    safe["items"] = [
        {
            key: item.get(key)
            for key in ("line_no", "sku", "title", "return_category", "qty", "unit_price_inr")
            if item.get(key) is not None
        }
        for item in order.get("items", [])
    ]
    if "is_plus" in order:
        safe["is_plus"] = order["is_plus"]
    return safe


def _business_day_add(start: date, days: int, holidays: set[date]) -> date:
    cursor = start
    remaining = days
    while remaining > 0:
        cursor += timedelta(days=1)
        if cursor.weekday() != 6 and cursor not in holidays:
            remaining -= 1
    return cursor


@dataclass
class ConversationState:
    conversation_id: str
    order_id: str | None = None
    verified_order: dict[str, Any] | None = None
    verification_attempts: int = 0
    awaiting_verification: bool = False
    verification_locked: bool = False
    greeted: bool = False
    turns: int = 0
    issue_summary: str = ""
    recent_context: list[str] = field(default_factory=list)


@dataclass
class TurnResult:
    reply: str
    decision: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply": self.reply,
            "decision": self.decision,
            "retrieved_evidence": self.evidence,
            "tool_calls": self.tool_calls,
        }


class ConversationAgent:
    """Stateful turn handler; the model never decides whether verification passed."""

    def __init__(
        self,
        text_agent: GroundedTextAgent,
        tools: CapstoneToolGateway,
        dataset_dir: str | Path,
    ):
        self.text_agent = text_agent
        self.tools = tools
        self.dataset_dir = Path(dataset_dir)
        self.holidays = self._load_holidays()

    def _load_holidays(self) -> set[date]:
        path = self.dataset_dir / "data" / "holidays.csv"
        with path.open(encoding="utf-8-sig", newline="") as source:
            return {date.fromisoformat(row["date"]) for row in csv.DictReader(source)}

    def _evidence_records(self, *section_ids: str) -> list[dict[str, Any]]:
        by_id = {
            item.evidence_id: item for item in self.text_agent.retriever.evidence
        }
        return [by_id[section_id].to_dict() for section_id in section_ids if section_id in by_id]

    def _greet_turn(self, state: ConversationState, result: TurnResult) -> TurnResult:
        first_reply = not state.greeted
        result.reply = self._greet(state, result.reply)
        if first_reply:
            known = {item.get("evidence_id") for item in result.evidence}
            result.evidence.extend(
                item for item in self._evidence_records("STY-1")
                if item.get("evidence_id") not in known
            )
        return result

    @staticmethod
    def _greet(state: ConversationState, reply: str) -> str:
        if state.greeted:
            return reply
        state.greeted = True
        if re.search(r"\bAI support assistant\b|\bAI assistant\b", reply[:180], re.I):
            return reply
        return f"Hi, I'm ZentroMart's AI support assistant. {reply}"

    def _ticket(
        self,
        state: ConversationState,
        message: str,
        *,
        reason: str,
        intent: str,
        checks: str = "No order details were disclosed before verification.",
    ) -> TurnResult:
        issue = _redact_proofs(message).strip()
        summary_parts = [f"Issue: {issue or state.issue_summary or intent}.", f"Intent: {intent}."]
        if state.order_id:
            summary_parts.append(f"Order ID: {state.order_id}.")
        if state.issue_summary and state.issue_summary not in issue:
            summary_parts.append(f"Earlier context: {state.issue_summary}.")
        summary_parts.extend((f"Checks: {checks}", f"Escalation reason: {reason}."))
        outcome = self.tools.create_ticket(
            " ".join(summary_parts),
            order_id=state.order_id,
            reason=reason,
            intent=intent,
        )
        call = {"tool": "create_ticket", "created": bool(outcome.get("created")), "reason": reason}
        if reason == "high_value_damage":
            evidence = self._evidence_records("DWM-3", "ESC-1", "ESC-4")
        elif reason == "refund_more_than_three_business_days_past_maximum":
            evidence = self._evidence_records("ESC-1", "REF-2", "REF-6")
        elif reason == "third_or_later_contact_within_seven_days":
            evidence = self._evidence_records("ESC-1", "ESC-3")
        else:
            evidence = self._evidence_records("ESC-1", "ESC-4")
        if outcome.get("created"):
            confirmation = "I’ve sent this to a human colleague for review. They’ll follow up through support."
            ticket_id = outcome.get("ticket_id")
            if ticket_id:
                confirmation += f" Your ticket reference is {ticket_id}."
            if reason == "refund_more_than_three_business_days_past_maximum":
                rrn = next((
                    state.verified_order.get(key)
                    for key in ("return_rrn", "cancellation_rrn", "rto_rrn")
                    if state.verified_order and state.verified_order.get(key)
                ), None)
                if rrn:
                    confirmation += f" Your refund reference number (RRN) is {rrn}."
            return TurnResult(
                reply=confirmation,
                decision="escalate",
                evidence=evidence,
                tool_calls=[call],
            )
        return TurnResult(
            reply="I couldn’t open a handover just now. Please contact ZentroMart support directly for help.",
            decision="escalate",
            evidence=evidence,
            tool_calls=[call],
        )

    def _refund_escalation_reason(self, order: dict[str, Any]) -> str | None:
        today = date(2026, 10, 20)
        timelines = {
            "upi": 3,
            "card": 7,
            "netbanking": 5,
            "wallet": 1,
            "cod_bank": 7,
            "cod_wallet": 1,
        }
        prefixes = ("return", "cancellation", "rto")
        for prefix in prefixes:
            started = order.get(f"{prefix}_refund_initiated_on")
            completed = order.get(f"{prefix}_refund_completed_on")
            method = order.get(f"{prefix}_refund_method")
            if not started or completed:
                continue
            maximum = timelines.get(str(method or "").casefold())
            if maximum is None:
                continue
            try:
                initiated = date.fromisoformat(str(started)[:10])
            except ValueError:
                continue
            overdue_after = _business_day_add(
                _business_day_add(initiated, maximum, self.holidays), 3, self.holidays
            )
            if today > overdue_after:
                return "refund_more_than_three_business_days_past_maximum"
        return None

    def _order_evidence(self, safe_order: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        order_id = str(safe_order.get("order_id", ""))
        source_id = f"orders.csv:{order_id}"
        text = "\n".join(
            f"{key}: {value}"
            for key, value in safe_order.items()
            if key not in {"order_id", "items"} and value not in (None, "")
        )
        for item in safe_order.get("items", []):
            text += "\nItem: " + ", ".join(f"{key}={value}" for key, value in item.items())
        evidence = {
            "evidence_id": source_id,
            "source_type": "verified_order_lookup",
            "source_id": source_id,
            "section_id": source_id,
            "title": "Verified order record",
            "text": text,
            "status": "verified for this conversation",
            "effective_from": "2026-10-20",
            "doc_type": "order_data",
            "score": 1.0,
        }
        return text, evidence

    def respond(self, state: ConversationState, customer_message: str) -> TurnResult:
        message = customer_message.strip()
        state.turns += 1
        if not message:
            return TurnResult("Sorry, I didn’t catch that. Could you say that again?", "clarify")

        if not re.search(r"last four|digits|pincode|pin code|registered phone|आख़िरी चार|आखिरी चार", message, re.I):
            state.recent_context.append(_redact_proofs(message))
            state.recent_context = state.recent_context[-4:]
        if DAMAGE_RE.search(message) or "refund" in message.casefold():
            state.issue_summary = _redact_proofs(message)[:300]

        found_order_id = _extract_order_id(message)

        # These conditions can be routed to a human without exposing order data.
        reason_intent = None
        if SAFETY_RE.search(message):
            reason_intent = ("product_safety_risk", "safety_risk")
        elif FRAUD_RE.search(message):
            reason_intent = ("unrecognized_account_activity", "fraud")
        elif LEGAL_RE.search(message):
            reason_intent = ("legal_or_consumer_complaint", "legal_complaint")
        elif HUMAN_RE.search(message):
            reason_intent = ("customer_requested_human", "human_requested")
        if reason_intent:
            if found_order_id and state.order_id is None:
                state.order_id = found_order_id
            result = self._ticket(
                state, message, reason=reason_intent[0], intent=reason_intent[1]
            )
            return self._greet_turn(state, result)

        if found_order_id and found_order_id != state.order_id:
            state.order_id = found_order_id
            state.verified_order = None
            state.verification_attempts = 0
            state.awaiting_verification = True
            state.verification_locked = False
            self.tools.lookup_order(found_order_id)
            reply = "For your privacy, please share either the last four digits of your registered phone number or the delivery pincode for this order."
            result = TurnResult(reply, "clarify", evidence=self._evidence_records("IDV-2"), tool_calls=[
                {"tool": "lookup_order", "status": "verification_required", "order_id": found_order_id}
            ])
            return self._greet_turn(state, result)

        order_specific_followup = bool(
            ORDER_HELP_RE.search(message) or ORDER_FOLLOWUP_RE.search(message)
        )
        if state.order_id and state.verified_order is None and state.verification_attempts < 2:
            proof = _extract_proof(message)
            if proof is None and not order_specific_followup:
                # A pending private lookup must not block unrelated public policy help.
                result = self.text_agent.respond(
                    message, conversation_context=state.recent_context
                )
                result = TurnResult(
                    reply=result.reply,
                    decision=result.decision,
                    evidence=result.retrieved_evidence,
                )
                return self._greet_turn(state, result)
            if not state.awaiting_verification or proof is None:
                reply = "Please provide the last four digits of your registered phone number or the delivery pincode for the order."
                return self._greet_turn(state, TurnResult(reply, "clarify", evidence=self._evidence_records("IDV-2")))
            state.verification_attempts += 1
            outcome = self.tools.lookup_order(state.order_id, verification_value=proof)
            # Do not put the submitted proof into state, tool logs, or the reply.
            if outcome.get("verified") is True and outcome.get("found") is True:
                state.verified_order = _safe_order(outcome)
                state.awaiting_verification = False
                call = {"tool": "lookup_order", "status": "verified", "order_id": state.order_id}
                substantive_query = re.sub(r"\b\d{4,6}\b", "", _ascii_digits(message)).strip()
                substantive_query = re.sub(
                    r"\b(last|four|digits?|registered|phone|number|pincode|pin|code|are|is|my|the)\b",
                    " ", substantive_query, flags=re.I,
                ).strip()
                if len(re.sub(r"\W", "", substantive_query)) <= 8:
                    earlier_request = " ".join(state.recent_context[-4:])
                    if re.search(r"where is (?:it|my order)|hasn.t arrived|not arrived|track my order|where is it", earlier_request, re.I):
                        result = self._answer_with_order(state, earlier_request, [call])
                        known = {item.get("evidence_id") for item in result.evidence}
                        result.evidence.extend(
                            item for item in self._evidence_records("IDV-2")
                            if item.get("evidence_id") not in known
                        )
                        return result
                    reply = "Thanks, I’ve verified the order. What would you like help with?"
                    return self._greet_turn(state, TurnResult(reply, "answer", evidence=self._evidence_records("IDV-2"), tool_calls=[call]))
                return self._answer_with_order(state, substantive_query, [call])
            if state.verification_attempts < 2:
                reply = "Those details didn’t match. Please try again once with the registered phone’s last four digits or the delivery pincode."
                return self._greet_turn(state, TurnResult(reply, "clarify", evidence=self._evidence_records("IDV-3"), tool_calls=[
                    {"tool": "lookup_order", "status": "verification_failed", "order_id": state.order_id}
                ]))
            state.awaiting_verification = False
            state.verification_locked = True
            reply = "I’m unable to verify this order, so I can’t share its details. Please contact support from the registered phone number; I can still answer general policy questions."
            return self._greet_turn(state, TurnResult(reply, "refuse", evidence=self._evidence_records("IDV-3"), tool_calls=[
                {"tool": "lookup_order", "status": "verification_failed", "order_id": state.order_id}
            ]))

        if state.order_id and state.verified_order is None and state.verification_attempts >= 2:
            personal_order_request = (
                ORDER_HELP_RE.search(message)
                or ORDER_FOLLOWUP_RE.search(message)
                or re.search(r"\b(my|this|that)\s+(order|refund|delivery|shipment)\b", message, re.I)
            )
            if personal_order_request or (state.verification_locked and _extract_proof(message)):
                reply = "I’m unable to verify this order, so I can’t share its details. Please contact support from the registered phone number."
                return TurnResult(reply, "refuse")

        if state.verified_order:
            order = state.verified_order
            if int(order.get("contacts_last_7_days") or 0) >= 2:
                return self._ticket(
                    state,
                    message,
                    reason="third_or_later_contact_within_seven_days",
                    intent="repeat_contact",
                    checks="Verified order lookup shows at least two earlier contacts in the last seven days.",
                )
            overdue_reason = (
                self._refund_escalation_reason(order)
                if re.search(r"refund|रिफ़ंड|रिफंड", message, re.I)
                else None
            )
            if overdue_reason:
                return self._ticket(
                    state,
                    message,
                    reason=overdue_reason,
                    intent="refund_status",
                    checks="Verified order lookup shows an initiated refund still incomplete beyond the escalation threshold.",
                )
            if DAMAGE_RE.search(message):
                high_value = any(int(item.get("unit_price_inr") or 0) > 10000 for item in order.get("items", []))
                if high_value:
                    return self._ticket(
                        state,
                        message,
                        reason="high_value_damage",
                        intent="damaged_item",
                        checks="Caller verified; order item price exceeds ₹10,000.",
                    )
        return self._answer_with_order(state, message, []) if state.verified_order else self._answer_or_clarify(state, message)

    def _answer_with_order(
        self,
        state: ConversationState,
        message: str,
        calls: list[dict[str, Any]],
    ) -> TurnResult:
        safe_order = state.verified_order or {}
        deterministic = self._answer_order_facts(state, message, calls)
        if deterministic is not None:
            return self._greet_turn(state, deterministic)
        context, order_evidence = self._order_evidence(safe_order) if safe_order else ("", {})
        result = self.text_agent.respond(
            message,
            extra_context=context,
            extra_evidence=[order_evidence] if order_evidence else [],
            conversation_context=state.recent_context,
        )
        return self._greet_turn(state, TurnResult(
            reply=result.reply,
            decision=result.decision,
            evidence=result.retrieved_evidence,
            tool_calls=calls,
        ))

    @staticmethod
    def _format_order_date(value: Any) -> str:
        try:
            parsed = date.fromisoformat(str(value)[:10])
        except (TypeError, ValueError):
            return str(value or "")
        return f"{parsed.day} {parsed.strftime('%B %Y')}"

    def _answer_order_facts(
        self,
        state: ConversationState,
        message: str,
        calls: list[dict[str, Any]],
    ) -> TurnResult | None:
        order = state.verified_order or {}
        normalized = message.casefold()

        if re.search(r"\b(cancel|cancellation|रद्द)\b|कैंसल", normalized):
            status = str(order.get("status", "")).casefold()
            if status in {"shipped", "out_for_delivery", "delivery_attempted"}:
                method = str(order.get("payment_method", "")).casefold()
                refund_days = {"upi": "1–3", "card": "5–7", "netbanking": "3–5"}.get(method)
                if str(order.get("payment_method", "")).casefold() == "cod":
                    timing = " For a cash-on-delivery order, nothing is charged if you refuse the parcel."
                else:
                    payment_timing = f" After initiation, the {method.upper()} refund takes {refund_days} business days." if refund_days else ""
                    timing = " For a prepaid order, the refund is initiated once the parcel returns to the warehouse, usually 7 to 10 calendar days after the last delivery attempt." + payment_timing
                reply = "I have to decline cancellation because this order has shipped. You can refuse the parcel at delivery." + timing
                return TurnResult(reply, "refuse", self._evidence_records("CAN-2", "REF-1"), calls)
            if status in {"placed", "confirmed", "processing"}:
                return TurnResult(
                    "This order has not shipped yet, so you can cancel it from My Orders in the app.",
                    "answer", self._evidence_records("CAN-1"), calls,
                )

        if any(term in normalized for term in ("voucher", "compensation", "compensate", "वाउचर", "मुआवज़ा", "मुआवजा")):
            if str(order.get("delivery_type", "")).casefold() == "standard":
                return TurnResult(
                    "I’m sorry your Standard delivery was late. We don’t offer compensation or vouchers for late Standard deliveries.",
                    "refuse", self._evidence_records("DEL-4", "ESC-4"), calls,
                )

        if re.search(r"not at home|away|out when|घर पर नहीं|घर पर नही", normalized):
            return TurnResult(
                "The courier makes up to 3 delivery attempts. After a missed attempt, you’ll receive an SMS to reschedule.",
                "answer", self._evidence_records("DEL-3"), calls,
            )

        if re.search(r"\b(ship|shipped|dispatch|dispatch(ed)?|sent)\b|भेजा", normalized) and re.search(r"when|kab|कब", normalized):
            shipped = order.get("shipped_on")
            if shipped:
                return TurnResult(
                    f"Your order shipped on {self._format_order_date(shipped)}.",
                    "answer", self._evidence_records("TRK-1"), calls,
                )

        if re.search(r"\b(where is (?:it|my order)|hasn.t arrived|not arrived|track my order|order status)\b|मेरा ऑर्डर कहाँ", normalized):
            promised = order.get("promised_date")
            status = str(order.get("status", "")).replace("_", " ")
            if promised:
                body = f"Your order is {status}. It is expected by {self._format_order_date(promised)}."
                return TurnResult(body, "answer", self._evidence_records("TRK-1"), calls)

        if any(term in normalized for term in ("refund", "रिफ़ंड", "रिफंड")):
            if str(order.get("return_status", "")).casefold() == "picked_up" and not order.get("return_qc_passed_on"):
                picked = self._format_order_date(order.get("return_pickup_on"))
                reply = f"The return was picked up on {picked} and is awaiting the warehouse quality check (QC). Once it passes, the refund is initiated within 2 business days of pickup."
                return TurnResult(reply, "answer", self._evidence_records("RET-7", "REF-1"), calls)
            method = str(order.get("payment_method", "")).casefold()
            if method:
                augmented = f"{message} refund by {method}"
                structured = self.text_agent.structured_data.answer_question(augmented)
                if structured:
                    evidence_ids = list(structured.evidence_ids) + ["RET-7"]
                    if order.get("is_plus"):
                        evidence_ids.append("PLS-2")
                    evidence = self._evidence_records(*evidence_ids)
                    reply = structured.reply
                    if order.get("is_plus"):
                        reply += " For Plus members, the refund is initiated at pickup."
                    return TurnResult(reply, structured.decision, evidence, calls)

        if any(term in normalized for term in ("return", "send back", "वापस", "लौट")):
            items = order.get("items", [])
            product = self.text_agent.structured_data._product_from_question(message)
            selected = next((item for item in items if product and item.get("sku") == product.get("sku")), None)
            if selected is None and items:
                selected = items[0]
            if selected:
                delivered = order.get("delivered_on")
                placed = str(order.get("placed_at", ""))[:10]
                augmented = f"{message} {selected.get('title', '')} delivered {delivered} ordered {placed}"
                structured = self.text_agent.structured_data.answer_question(augmented)
                if structured:
                    evidence_ids = list(structured.evidence_ids)
                    reply = structured.reply
                    if selected.get("return_category") == "fashion":
                        evidence_ids.append("RET-6")
                        if order.get("is_plus"):
                            reply += " The ₹49 reverse-pickup fee is waived for Plus members."
                        else:
                            reply += " A ₹49 reverse-pickup fee applies to change-of-mind returns."
                    return TurnResult(
                        reply, structured.decision,
                        self._evidence_records(*evidence_ids), calls,
                    )

        return None

    def _answer_or_clarify(self, state: ConversationState, message: str) -> TurnResult:
        if ORDER_HELP_RE.search(message) and not _extract_order_id(message):
            reply = "I can help with that. What’s the order ID?"
            return self._greet_turn(state, TurnResult(reply, "clarify"))
        result = self.text_agent.respond(message, conversation_context=state.recent_context)
        return self._greet_turn(state, TurnResult(
            reply=result.reply,
            decision=result.decision,
            evidence=result.retrieved_evidence,
        ))
