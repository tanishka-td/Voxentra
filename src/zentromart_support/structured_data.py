"""Deterministic lookups and policy calculations over the capstone CSV data."""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any


REFERENCE_DATE = date(2026, 10, 20)
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7,
    "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    "जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4,
    "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8, "सितंबर": 9,
    "अक्टूबर": 10, "नवंबर": 11, "दिसंबर": 12,
}
RETURN_DAYS = {
    "electronics": 7,
    "fashion": 15,
    "home_kitchen": 10,
    "beauty_sealed": 7,
    "books_media": 10,
}
NON_RETURNABLE = {"grocery", "innerwear", "digital"}
REFUND_DAYS = {
    "upi": (1, 3),
    "card": (5, 7),
    "netbanking": (3, 5),
    "cod_bank": (5, 7),
}


@dataclass(frozen=True)
class StructuredAnswer:
    reply: str
    evidence_ids: tuple[str, ...]
    decision: str = "answer"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as source:
        return [dict(row) for row in csv.DictReader(source)]


def _ascii_digits(value: str) -> str:
    return "".join(
        str(unicodedata.digit(char)) if unicodedata.category(char) == "Nd" else char
        for char in value
    )


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = "".join(
        char if char.isalnum() or unicodedata.category(char).startswith("M") else " "
        for char in normalized
    )
    return " ".join(tokens.split())


def _parse_dates(text: str) -> list[date]:
    value = _ascii_digits(text)
    found: list[date] = []
    for match in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", value):
        try:
            found.append(date(int(match[1]), int(match[2]), int(match[3])))
        except ValueError:
            continue
    month_pattern = "|".join(sorted((re.escape(name) for name in MONTHS), key=len, reverse=True))
    pattern = re.compile(
        rf"(?<!\w)(\d{{1,2}})\s+({month_pattern})(?:\s+(20\d{{2}}))?(?!\w)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(value):
        month = MONTHS.get(match[2].casefold(), MONTHS.get(match[2]))
        if month is None:
            continue
        try:
            found.append(date(int(match[3] or REFERENCE_DATE.year), month, int(match[1])))
        except ValueError:
            continue
    return sorted(set(found))


def add_business_days(start: date, count: int, holidays: set[date]) -> date:
    """Count Monday-Saturday as business days, excluding listed holidays."""
    current = start
    remaining = count
    while remaining:
        current += timedelta(days=1)
        if current.weekday() != 6 and current not in holidays:
            remaining -= 1
    return current


class StructuredSupportData:
    """Read-only product, pincode and holiday data with deterministic rules."""

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.products = _read_csv(self.dataset_dir / "data" / "products.csv")
        self.pincodes = {
            row["pincode"]: row
            for row in _read_csv(self.dataset_dir / "data" / "pincodes.csv")
            if row.get("pincode")
        }
        self.holidays = {
            date.fromisoformat(row["date"])
            for row in _read_csv(self.dataset_dir / "data" / "holidays.csv")
            if row.get("date")
        }
        self._products_by_sku = {
            row["sku"].strip(): row for row in self.products if row.get("sku", "").strip()
        }

    def _product_from_question(self, question: str) -> dict[str, str] | None:
        sku_match = re.search(r"\b[A-Za-z]{3}-\d{4}\b", question)
        if sku_match:
            return self._products_by_sku.get(sku_match.group(0).upper())
        normalized_question = _normalize(question)
        matches: list[tuple[int, dict[str, str]]] = []
        for row in self.products:
            title = row.get("title", "")
            # Customers often omit the color or finish after a comma.
            variants = (title, title.split(",", 1)[0])
            for variant in variants:
                normalized_title = _normalize(variant)
                if normalized_title and normalized_title in normalized_question:
                    matches.append((len(normalized_title), row))
                    break
        if not matches:
            return None
        best_length = max(length for length, _ in matches)
        best = [row for length, row in matches if length == best_length]
        if len(best) == 1:
            return best[0]
        return None

    def _language(self, question: str) -> str:
        if any("DEVANAGARI" in unicodedata.name(char, "") for char in question):
            return "hi"
        tokens = set(_normalize(question).split())
        return "hi-en" if tokens.intersection({"hai", "hain", "mein", "mujhe", "kitne", "wala", "ko", "kya", "pe"}) else "en"

    def _greet(self, question: str, body: str) -> str:
        lang = self._language(question)
        if lang == "hi":
            return f"नमस्ते, मैं ZentroMart का AI सहायक हूँ। {body}"
        if lang == "hi-en":
            return f"Namaste, main ZentroMart ka AI assistant hoon. {body}"
        return f"I’m ZentroMart’s AI support assistant. {body}"

    def _refund_answer(self, question: str) -> StructuredAnswer | None:
        normalized = _normalize(question)
        if not any(term in normalized for term in ("refund", "रिफंड", "वापसी", "money back", "paise")):
            return None
        is_cod = "cash on delivery" in normalized or re.search(r"\bcod\b", normalized) is not None
        cod_wallet = "wallet" in normalized
        cod_bank = "bank transfer" in normalized or "neft" in normalized
        if is_cod and (not (cod_wallet or cod_bank) or (cod_wallet and cod_bank)):
            body = "For a COD refund, you can choose ZentroMart Wallet (instant, within 2 hours) or a bank transfer (5–7 business days after bank details are verified)."
            return StructuredAnswer(self._greet(question, body), ("REF-2", "REF-3"))
        method: str | None = None
        if "net banking" in normalized or "netbanking" in normalized:
            method = "netbanking"
        elif "upi" in normalized:
            method = "upi"
        elif "credit card" in normalized or "debit card" in normalized or re.search(r"\bcard\b", normalized):
            method = "card"
        elif is_cod and ("neft" in normalized or "bank transfer" in normalized):
            method = "cod_bank"
        elif "wallet" in normalized:
            body = "Wallet refunds are instant, within 2 hours."
            if self._language(question) == "hi":
                body = "Wallet refund तुरंत, 2 घंटे के भीतर आता है।"
            elif self._language(question) == "hi-en":
                body = "Wallet refund turant, 2 hours ke andar aa jata hai."
            return StructuredAnswer(self._greet(question, body), ("REF-2",))

        # COD handling above preserves the choice of refund method.
        if method is None:
            return None

        if method in REFUND_DAYS:
            minimum, maximum = REFUND_DAYS[method]
            payment_name = {
                "upi": "UPI", "card": "credit or debit card",
                "netbanking": "net banking", "cod_bank": "bank transfer",
            }[method]
            body = f"{payment_name.capitalize()} refunds take {minimum}–{maximum} business days after initiation."
            dates = _parse_dates(question)
            evidence_ids = ["REF-2"]
            if dates:
                latest = add_business_days(dates[-1], maximum, self.holidays)
                body += f" Using the dataset calendar, it should arrive by {latest.day} {latest.strftime('%B %Y')}."
                evidence_ids.extend(
                    f"holidays.csv:{holiday.isoformat()}"
                    for holiday in sorted(self.holidays)
                    if dates[-1] < holiday <= latest
                )
            if self._language(question) == "hi":
                body = f"{payment_name.capitalize()} refund शुरू होने के बाद {minimum}–{maximum} कार्यदिवस लेता है।"
                if dates:
                    body += f" Dataset calendar के अनुसार, अधिकतम {latest.day} {latest.strftime('%B %Y')} तक आ जाना चाहिए।"
            elif self._language(question) == "hi-en":
                body = f"{payment_name.capitalize()} refund shuru hone ke baad {minimum}–{maximum} business days leta hai."
                if dates:
                    body += f" Dataset calendar ke hisaab se, maximum {latest.day} {latest.strftime('%B %Y')} tak aa jana chahiye."
            return StructuredAnswer(self._greet(question, body), tuple(evidence_ids))
        return None

    def _warranty_answer(self, question: str, product: dict[str, str]) -> StructuredAnswer:
        sku = product["sku"]
        seller = product.get("fulfilled_by", "").strip()
        months = product.get("warranty_months", "").strip()
        brand_warranty = (
            f"a {months}-month brand warranty"
            if months.isdigit() and int(months) > 0
            else "no brand-warranty period listed in the catalog"
        )
        assured = product.get("assured", "").strip().casefold() == "true"
        evidence = [f"products.csv:{sku}", "WAR-1"]
        if seller.casefold() != "zentromart":
            detail = f"This listing has {brand_warranty}; ZentroMart Assured does not apply to marketplace-seller items."
            evidence.append("MKS-3")
        elif assured:
            detail = f"This product has a 12-month ZentroMart Assured warranty from delivery, plus {brand_warranty}. Claims are raised in the app under My Orders."
            evidence.append("WAR-2")
        else:
            detail = f"This listing has {brand_warranty}. Brand warranty claims go through the manufacturer’s service centre."
        return StructuredAnswer(self._greet(question, detail), tuple(evidence))

    def _return_answer(self, question: str, product: dict[str, str] | None) -> StructuredAnswer | None:
        normalized = _normalize(question)
        terms = ("return", "send back", "लौट", "वापस")
        is_fee = any(term in normalized for term in ("pickup fee", "collection fee", "return fee", "charged", "charge", "collection", "pickup"))
        if not any(term in normalized for term in terms) and not is_fee:
            return None
        category = product.get("return_category", "").strip() if product else ""
        title_query = normalized
        if product is None:
            for candidate, keywords in {
                "electronics": ("electronics", "electronic", "फोन", "मोबाइल"),
                "fashion": ("fashion", "clothing", "clothes", "shoes", "jeans", "bag", "kurta", "जूते", "कपड़े"),
                "home_kitchen": ("home", "kitchen", "furniture", "bedsheet", "घर"),
                "books_media": ("book", "stationery", "notebook", "books"),
                "beauty_sealed": ("beauty", "makeup", "lipstick", "perfume", "कॉस्मेटिक"),
                "grocery": ("grocery", "fresh", "perishable"),
                "innerwear": ("innerwear", "lingerie", "briefs", "swimsuit", "underwear"),
                "digital": ("gift card", "digital code"),
            }.items():
                if any(keyword in normalized for keyword in keywords):
                    category = candidate
                    break

        if is_fee:
            is_plus = bool(re.search(r"\bplus\b|member|प्लस", normalized))
            is_damaged = any(term in normalized for term in ("damaged", "torn", "wrong", "defective", "खराब"))
            if category == "fashion" and (is_plus or is_damaged):
                reason = "ZentroMart Plus members" if is_plus else "damaged, defective, or wrong items"
                return StructuredAnswer(
                    self._greet(question, f"There is no reverse-pickup fee for {reason}; the ₹49 fee is waived."),
                    ("RET-6",),
                )
            if category == "fashion":
                return StructuredAnswer(self._greet(question, "A ₹49 reverse-pickup fee applies to change-of-mind returns of clothing, footwear, and bags. It is waived for Plus members and damaged, defective, or wrong items."), ("RET-6",))
            return None

        if not category:
            body = "Which product or category is this about? Return windows differ by category."
            return StructuredAnswer(self._greet(question, body), ("RET-2",), "clarify")

        evidence = ["RET-2"]
        if product:
            evidence.insert(0, f"products.csv:{product['sku']}")
        if category in NON_RETURNABLE:
            label = {"grocery": "Grocery and fresh or perishable items", "innerwear": "Innerwear, lingerie, and swimwear", "digital": "Gift cards and digital codes"}[category]
            return StructuredAnswer(self._greet(question, f"{label} cannot be returned for change of mind."), tuple(evidence + ["RET-5"]))
        if category == "beauty_sealed":
            opened = any(term in normalized for term in ("opened", "open it", "tried it", "seal broken", "खोल", "खुला"))
            if opened:
                body = "Beauty and personal-care items are returnable only when unopened with the seal intact, so an opened item is not eligible for a change-of-mind return."
                if self._language(question) == "hi":
                    body = "Beauty और personal-care उत्पाद केवल तभी लौटाए जा सकते हैं जब वे बंद हों और seal intact हो। खोला हुआ उत्पाद change-of-mind return के लिए eligible नहीं है।"
                return StructuredAnswer(self._greet(question, body), tuple(evidence + ["RET-4", "RET-5"]))
        if category == "electronics":
            evidence.append("RET-3")

        delivery_dates = _parse_dates(question) if "deliver" in normalized or "डिलीवर" in normalized else []
        placed_text = question
        place_marker = re.search(r"\b(?:ordered|bought|sale|order kiya)\b|खरीदा|ऑर्डर किया", question, re.I)
        if place_marker:
            placed_text = question[place_marker.end():]
        placed_dates = _parse_dates(placed_text) if place_marker or any(term in normalized for term in ("sale", "bought", "ordered", "order kiya", "खरीदा", "ऑर्डर किया")) else []
        eligibility_query = any(term in normalized for term in ("still eligible", "can i return", "can return it", "return it now", "eligible for return")) or bool(
            re.search(r"(?:अब भी|अभी भी).*(?:वापस|लौट)|(?:वापस|लौट).*(?:कर सकती|कर सकता|करूं|करूँ|योग्य)|return kar sak(?:ti|ta|te)", normalized)
        )
        if category == "fashion" and placed_dates:
            placed = placed_dates[-1]
            if date(2026, 10, 1) <= placed <= date(2026, 10, 15):
                evidence.append("RET-8")
                delivery_dates = _parse_dates(question)
                body = "This fashion item has a 30-day return window from delivery because the order was placed during the festive sale."
                if eligibility_query and delivery_dates:
                    deadline = delivery_dates[-1] + timedelta(days=30)
                    body += f" The return window ends on {deadline.day} {deadline.strftime('%B %Y')}."
                return StructuredAnswer(self._greet(question, body), tuple(evidence))

        days = RETURN_DAYS.get(category)
        if category == "electronics" and delivery_dates and delivery_dates[-1] < date(2026, 8, 1):
            days = 10
        if days is None:
            return None
        if eligibility_query and delivery_dates:
            delivered = delivery_dates[-1]
            deadline = delivered + timedelta(days=days)
            if REFERENCE_DATE < delivered:
                return StructuredAnswer(
                    self._greet(question, "I can’t assess return eligibility before the delivery date. Please check the delivery date shown in My Orders."),
                    tuple(evidence),
                    "clarify",
                )
            state = "still within" if REFERENCE_DATE <= deadline else "outside"
            body = f"Based on the dataset reference date, this item is {state} its {days}-day return window, which ends on {deadline.day} {deadline.strftime('%B %Y')}."
            return StructuredAnswer(self._greet(question, body), tuple(evidence))
        category_name = {
            "electronics": "Electronics", "fashion": "Clothing, footwear, and bags",
            "home_kitchen": "Home, kitchen, and furniture", "beauty_sealed": "Beauty and personal care",
            "books_media": "Books and stationery",
        }[category]
        body = f"{category_name} have a {days}-day return window from the delivery date."
        if category == "beauty_sealed":
            body += " Beauty items must be unopened with the seal intact."
            evidence.append("RET-4")
        if self._language(question) == "hi":
            body = f"{category_name} के लिए delivery date से {days} दिन की return window है।"
        elif self._language(question) == "hi-en":
            body = f"{category_name} ke liye delivery date se {days} din ki return window hai."
        return StructuredAnswer(self._greet(question, body), tuple(evidence))

    def _pincode_answer(self, question: str) -> StructuredAnswer | None:
        normalized = _normalize(question)
        if not any(term in normalized for term in ("deliver", "delivery", "serviceable", "cod", "cash on delivery", "express")):
            return None
        digits = _ascii_digits(question)
        match = re.search(r"(?<!\d)\d{6}(?!\d)", digits)
        if not match:
            return None
        pincode = match.group(0)
        row = self.pincodes.get(pincode)
        if row is None:
            return StructuredAnswer(self._greet(question, "I can’t confirm service for that pincode from the available serviceability data."), ("DEL-5",), "escalate")
        serviceable = row.get("serviceable", "").casefold() == "true"
        city = row.get("city", "")
        body = f"{city} {pincode} is {'serviceable' if serviceable else 'not serviceable'}."
        if serviceable and ("cod" in normalized or "cash on delivery" in normalized):
            body += f" Cash on delivery is {'available' if row.get('cod_available', '').casefold() == 'true' else 'not available'} there."
        if serviceable and "express" in normalized:
            body += f" Express delivery is {'available' if row.get('express_available', '').casefold() == 'true' else 'not available'} there."
        return StructuredAnswer(self._greet(question, body), (f"pincodes.csv:{pincode}", "DEL-5"))

    def answer_question(self, question: str) -> StructuredAnswer | None:
        """Answer only when structured fields and deterministic rules are sufficient."""
        normalized = _normalize(question)
        product = self._product_from_question(question)
        if product and "warranty" in normalized:
            return self._warranty_answer(question, product)
        refund = self._refund_answer(question)
        if refund is not None:
            return refund
        return_answer = self._return_answer(question, product)
        if return_answer is not None:
            return return_answer
        return self._pincode_answer(question)
