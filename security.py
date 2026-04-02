"""
core/security.py — LLM Security, Guardrails & PII Protection
=============================================================
Features:
  • PII detection & masking (regex-based, zero external calls)
  • Prompt injection detection
  • Harmful content policy check
  • Input / output sanitisation
  • Token budget enforcement
  • Structured GuardrailResult for clear pass/fail reporting
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from core.logger import get_logger

logger = get_logger("security")


# ── PII Patterns ──────────────────────────────────────────────────────────────

PII_PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL":        re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
    "PHONE_US":     re.compile(
        r"\b(\+1[\s\-]?)?(\(?\d{3}\)?[\s\-]?)\d{3}[\s\-]?\d{4}\b"),
    "SSN":          re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "CREDIT_CARD":  re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|"
        r"3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"),
    "IP_ADDRESS":   re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "DATE_OF_BIRTH": re.compile(
        r"\b(?:dob|date of birth|born on)[:\s]+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
        re.IGNORECASE),
    "PASSPORT":     re.compile(
        r"\b[A-Z]{1,2}[0-9]{6,9}\b"),
    "AADHAAR":      re.compile(       # Indian national ID
        r"\b[2-9]{1}[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b"),
    "PAN":          re.compile(       # Indian PAN card
        r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"),
}

# ── Prompt Injection Signatures ───────────────────────────────────────────────

INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"you are now (a )?(?!an? AI)", re.IGNORECASE),
    re.compile(r"(pretend|act|roleplay|simulate).{0,30}(you are|as if)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN mode", re.IGNORECASE),
    re.compile(r"do anything now", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[\/INST\]"),
    re.compile(r"forget (your |all )?(instructions|training)", re.IGNORECASE),
]

# ── Harmful Content Keywords ──────────────────────────────────────────────────

HARMFUL_KEYWORDS: List[str] = [
    "generate malware", "write a virus", "hack into", "sql injection attack",
    "ddos attack", "exploit vulnerability", "bypass security",
    "synthesize [a-z]+ drug", "make explosives", "manufacture weapons",
]
HARMFUL_PATTERNS: List[re.Pattern] = [
    re.compile(kw, re.IGNORECASE) for kw in HARMFUL_KEYWORDS
]


# ── Result Types ──────────────────────────────────────────────────────────────

@dataclass
class PIIFinding:
    pii_type: str
    original:  str
    masked:    str
    position:  Tuple[int, int]


@dataclass
class GuardrailResult:
    passed:         bool                  = True
    violations:     List[str]             = field(default_factory=list)
    pii_findings:   List[PIIFinding]      = field(default_factory=list)
    sanitised_text: Optional[str]         = None

    def fail(self, reason: str):
        self.passed = False
        self.violations.append(reason)
        logger.warning(f"Guardrail violation: {reason}")


# ── Guardrails Engine ─────────────────────────────────────────────────────────

class GuardrailsEngine:
    """
    Run all security checks on text before it reaches the LLM (input)
    and after it returns (output).
    """

    def __init__(
        self,
        enable_pii:        bool = True,
        enable_injection:  bool = True,
        enable_harmful:    bool = True,
        mask_char:         str  = "*",
        max_input_chars:   int  = 8000,
    ):
        self.enable_pii       = enable_pii
        self.enable_injection = enable_injection
        self.enable_harmful   = enable_harmful
        self.mask_char        = mask_char
        self.max_input_chars  = max_input_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def check_input(self, text: str) -> GuardrailResult:
        """Validate & sanitise user / document input before LLM call."""
        result = GuardrailResult(sanitised_text=text)

        if not text or not text.strip():
            result.fail("Empty input")
            return result

        if len(text) > self.max_input_chars:
            result.fail(
                f"Input too long ({len(text)} chars > {self.max_input_chars})"
            )
            return result

        if self.enable_injection:
            self._check_injection(text, result)

        if self.enable_harmful:
            self._check_harmful(text, result)

        if self.enable_pii:
            sanitised, findings = self._detect_and_mask_pii(text)
            result.pii_findings  = findings
            result.sanitised_text = sanitised
            if findings:
                logger.info(
                    f"PII masked: {[f.pii_type for f in findings]}"
                )

        return result

    def check_output(self, text: str) -> GuardrailResult:
        """Validate LLM output before returning to user — mask leaked PII."""
        result = GuardrailResult(sanitised_text=text)

        if self.enable_pii:
            sanitised, findings = self._detect_and_mask_pii(text)
            result.pii_findings   = findings
            result.sanitised_text = sanitised
            if findings:
                logger.warning(
                    f"PII found in LLM OUTPUT — masked: "
                    f"{[f.pii_type for f in findings]}"
                )

        return result

    def mask_pii(self, text: str) -> str:
        """Quick helper — returns masked text only."""
        masked, _ = self._detect_and_mask_pii(text)
        return masked

    # ── Internal Checks ───────────────────────────────────────────────────────

    def _check_injection(self, text: str, result: GuardrailResult):
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                result.fail(
                    f"Prompt injection detected: pattern '{pattern.pattern}'"
                )

    def _check_harmful(self, text: str, result: GuardrailResult):
        for pattern in HARMFUL_PATTERNS:
            if pattern.search(text):
                result.fail(
                    f"Harmful content detected: pattern '{pattern.pattern}'"
                )

    def _detect_and_mask_pii(
        self, text: str
    ) -> Tuple[str, List[PIIFinding]]:
        findings: List[PIIFinding] = []
        masked_text = text

        for pii_type, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(masked_text):
                original = match.group()
                # Keep first 2 and last 2 chars visible, mask the rest
                if len(original) > 4:
                    replacement = (
                        original[:2]
                        + self.mask_char * (len(original) - 4)
                        + original[-2:]
                    )
                else:
                    replacement = self.mask_char * len(original)

                findings.append(PIIFinding(
                    pii_type=pii_type,
                    original=original,
                    masked=replacement,
                    position=match.span(),
                ))
                masked_text = masked_text.replace(original, replacement, 1)

        return masked_text, findings


# ── Module singleton ──────────────────────────────────────────────────────────
guardrails = GuardrailsEngine()
