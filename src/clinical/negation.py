"""
Negation detection for clinical NER.

Implements a rule-based NegEx/ConText-style algorithm to determine whether
recognised medical entities are affirmed or negated in context. This is
critical for medical coding: "no fever" should NOT be coded as a fever
diagnosis, while "fever" should.

Based on:
- Chapman et al. (2001) "A Simple Algorithm for Identifying Negated Findings"
- Harkema et al. (2009) "ConText: An Algorithm for Determining Negation,
  Experiencer, and Temporality of Clinical Conditions from Clinical Reports"

Extended with patterns commonly found in modern EHR documentation.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NegationStatus(Enum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    POSSIBLE = "possible"
    HYPOTHETICAL = "hypothetical"
    HISTORICAL = "historical"
    FAMILY = "family"


@dataclass
class NegationScope:
    """Represents a negation trigger and its scope in the text."""
    trigger_text: str
    trigger_start: int
    trigger_end: int
    scope_start: int
    scope_end: int
    status: NegationStatus
    direction: str  # "forward", "backward", or "bidirectional"


# ---------------------------------------------------------------------------
# Negation trigger lexicons
# ---------------------------------------------------------------------------

# Pre-negation triggers: negate entities that follow
_PRE_NEGATION_TRIGGERS = [
    # Standard negation
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bw/o\b",
    r"\babsence of\b",
    r"\bdeny\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bdenying\b",
    r"\bnegative for\b",
    r"\bneg\s+for\b",
    r"\bneg\.\s+for\b",
    r"\bnever\b",
    r"\bno evidence of\b",
    r"\bno e/o\b",
    r"\bno sign of\b",
    r"\bno signs of\b",
    r"\bno symptoms of\b",
    r"\bno complaint of\b",
    r"\bno complaints of\b",
    r"\bnot demonstrate\b",
    r"\bfails to reveal\b",
    r"\bfailed to reveal\b",
    r"\bnon\b[-\s]",
    r"\brunremarkable\b",
    r"\bfree of\b",
    r"\bresolved\b",
    r"\bhas been ruled out\b",
    r"\bruled out\b",
    r"\br/o\b",
    r"\brule out\b",
    r"\brules out\b",
    r"\bruled him out\b",
    r"\bruled her out\b",
    r"\bruled the patient out\b",
    r"\bno longer\b",
    r"\bno further\b",
    r"\bnot had\b",
    r"\bnot have\b",
    r"\bnot having\b",
    r"\bno history of\b",
    r"\bno h/o\b",
    r"\bno hx of\b",
    r"\bno known\b",
    r"\bwithout any\b",
    r"\bwithout evidence of\b",
    r"\bno longer has\b",
    r"\bno acute\b",
    r"\bno new\b",
    r"\bno significant\b",
    r"\bno appreciable\b",
    r"\bno obvious\b",
    r"\bno definite\b",
    r"\bno suspicious\b",
    r"\bno recurrence of\b",
    r"\bno recurrent\b",
    r"\bnot associated with\b",
    r"\bnot complain of\b",
    r"\bnot feel\b",
    r"\bnot experience\b",
    r"\bnot exhibit\b",
    r"\bnot reveal\b",
    r"\bnot show\b",
    r"\bfree from\b",
    r"\bclear of\b",
    r"\bnot suspicious\b",
    r"\btest negative\b",
    r"\btested negative\b",
]

# Post-negation triggers: negate entities that precede
_POST_NEGATION_TRIGGERS = [
    r"\brunlikely\b",
    r"\bhas been ruled out\b",
    r"\bwas ruled out\b",
    r"\bis ruled out\b",
    r"\bare ruled out\b",
    r"\bwere ruled out\b",
    r"\bnot found\b",
    r"\bnot seen\b",
    r"\bnot identified\b",
    r"\bnot detected\b",
    r"\bnot demonstrated\b",
    r"\bnot observed\b",
    r"\bnot present\b",
    r"\bnot appreciated\b",
    r"\bnot noted\b",
    r"\bnegative\b",
    r"\bfree\b",
    r"\babsent\b",
    r"\bwas negative\b",
    r"\bwere negative\b",
    r"\bis negative\b",
    r"\bare negative\b",
]

# Pseudo-negation triggers: look like negation but are NOT
# These override true negation when they appear
_PSEUDO_NEGATION_TRIGGERS = [
    r"\bno increase\b",
    r"\bno change\b",
    r"\bnot only\b",
    r"\bnot necessarily\b",
    r"\bnot certain\b",
    r"\bwithout difficulty\b",
    r"\bwithout hesitation\b",
    r"\bnot cause\b",
    r"\bgram negative\b",
    r"\bgram-negative\b",
    r"\bnot drain\b",
    r"\bnot extend\b",
    r"\bno suspicious change\b",
    r"\brule him in\b",
    r"\brule her in\b",
    r"\brule the patient in\b",
    r"\bnot insignificant\b",
    r"\bnot unlikely\b",
]

# Scope terminators: stop the negation scope from extending further
_SCOPE_TERMINATORS = [
    r"\bbut\b",
    r"\bhowever\b",
    r"\bthough\b",
    r"\balthough\b",
    r"\baside from\b",
    r"\bexcept\b",
    r"\bapart from\b",
    r"\bnevertheless\b",
    r"\byet\b",
    r"\bstill\b",
    r"\bwhich\b",
    r"\bwho\b",
    r"\bcause(s|d)?\b",
    r"\bleading to\b",
    r"\bsecondary to\b",
    r"\breason for\b",
    r"\breturn(s|ed)?\b",
    # Section boundaries
    r"\b(assessment|plan|impression|diagnosis|allergies)\s*:",
    r"\n\s*\n",  # blank lines
    r"[.;]",  # sentence boundaries
]

# Uncertainty / possibility triggers
_POSSIBLE_TRIGGERS = [
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bprobable\b",
    r"\bprobably\b",
    r"\bsuspect(s|ed)?\b",
    r"\bquestionable\b",
    r"\bsuggestive of\b",
    r"\bconsistent with\b",
    r"\bconcerning for\b",
    r"\bcannot be excluded\b",
    r"\bcannot rule out\b",
    r"\bcannot exclude\b",
    r"\bdifferential\b",
    r"\bddx\b",
    r"\bmay have\b",
    r"\bmight have\b",
    r"\bcould have\b",
    r"\blikely\b",
    r"\bappears\b",
]

# Historical context triggers
_HISTORICAL_TRIGGERS = [
    r"\bhistory of\b",
    r"\bh/o\b",
    r"\bhx of\b",
    r"\bpast medical history\b",
    r"\bpmh\b",
    r"\bpmhx\b",
    r"\bprevious(ly)?\b",
    r"\bprior\b",
    r"\bformer(ly)?\b",
    r"\bremote\b",
]

# Family context triggers
_FAMILY_TRIGGERS = [
    r"\bfamily history of\b",
    r"\bfhx of\b",
    r"\bfhx\b",
    r"\bfamily hx\b",
    r"\b(mother|father|sister|brother|parent|sibling|grandmother|grandfather)\b.*\bhad\b",
    r"\b(maternal|paternal)\b.*\bhistory\b",
]


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


class NegationDetector:
    """
    Detects negation, possibility, historical, and family context for
    medical entities in clinical text.

    Uses a ConText-style algorithm with forward and backward scope
    propagation, terminated by scope-breaking cues.
    """

    def __init__(self, scope_window: int = 6):
        """
        Parameters
        ----------
        scope_window : int
            Maximum number of words the negation scope extends
            from the trigger (in the trigger's direction).
        """
        self.scope_window = scope_window

        self._pre_neg = _compile_patterns(_PRE_NEGATION_TRIGGERS)
        self._post_neg = _compile_patterns(_POST_NEGATION_TRIGGERS)
        self._pseudo_neg = _compile_patterns(_PSEUDO_NEGATION_TRIGGERS)
        self._terminators = _compile_patterns(_SCOPE_TERMINATORS)
        self._possible = _compile_patterns(_POSSIBLE_TRIGGERS)
        self._historical = _compile_patterns(_HISTORICAL_TRIGGERS)
        self._family = _compile_patterns(_FAMILY_TRIGGERS)

        logger.info(
            "NegationDetector initialised (scope_window=%d, %d pre-neg, %d post-neg triggers).",
            scope_window, len(self._pre_neg), len(self._post_neg),
        )

    def _find_triggers(
        self, text: str, patterns: List[re.Pattern], direction: str, status: NegationStatus,
    ) -> List[NegationScope]:
        """Find all trigger matches in text and compute their scopes."""
        triggers = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                # Check this isn't a pseudo-negation
                is_pseudo = False
                for pseudo in self._pseudo_neg:
                    pm = pseudo.search(text, max(0, match.start() - 10), match.end() + 10)
                    if pm and pm.start() <= match.start() and pm.end() >= match.end():
                        is_pseudo = True
                        break
                if is_pseudo:
                    continue

                scope_start, scope_end = self._compute_scope(
                    text, match.start(), match.end(), direction,
                )
                triggers.append(NegationScope(
                    trigger_text=match.group(0),
                    trigger_start=match.start(),
                    trigger_end=match.end(),
                    scope_start=scope_start,
                    scope_end=scope_end,
                    status=status,
                    direction=direction,
                ))
        return triggers

    def _compute_scope(
        self, text: str, trigger_start: int, trigger_end: int, direction: str,
    ) -> Tuple[int, int]:
        """
        Compute the character-level scope of a negation trigger.

        Scope extends from the trigger in the given direction, limited by:
        - scope_window (number of words)
        - Scope terminators (conjunctions, section boundaries)
        - End of text
        """
        if direction == "forward":
            scope_text = text[trigger_end:]
            # Find the nearest terminator
            terminator_pos = len(scope_text)
            for term_pattern in self._terminators:
                m = term_pattern.search(scope_text)
                if m and m.start() < terminator_pos:
                    terminator_pos = m.start()

            # Also limit by word count
            words = scope_text[:terminator_pos].split()
            if len(words) > self.scope_window:
                # Find character position of the Nth word boundary
                word_end = 0
                for i, word in enumerate(words[:self.scope_window]):
                    word_end = scope_text.index(word, word_end) + len(word)
                terminator_pos = min(terminator_pos, word_end)

            return trigger_start, trigger_end + terminator_pos

        elif direction == "backward":
            scope_text = text[:trigger_start]
            # Find the nearest terminator (searching backward)
            terminator_pos = 0
            for term_pattern in self._terminators:
                for m in term_pattern.finditer(scope_text):
                    if m.end() > terminator_pos:
                        terminator_pos = m.end()

            # Limit by word count
            words = scope_text[terminator_pos:].split()
            if len(words) > self.scope_window:
                word_start = len(scope_text)
                for i, word in enumerate(reversed(words[-self.scope_window:])):
                    idx = scope_text.rfind(word, terminator_pos, word_start)
                    if idx >= 0:
                        word_start = idx
                terminator_pos = max(terminator_pos, word_start)

            return terminator_pos, trigger_end

        else:  # bidirectional
            fwd = self._compute_scope(text, trigger_start, trigger_end, "forward")
            bwd = self._compute_scope(text, trigger_start, trigger_end, "backward")
            return bwd[0], fwd[1]

    def detect(
        self,
        text: str,
        entities: Optional[List[Dict]] = None,
    ) -> List[NegationScope]:
        """
        Detect all negation/context scopes in a text.

        Parameters
        ----------
        text : str
            Clinical text.
        entities : list of dict, optional
            If provided, only return scopes that overlap with entities.
            Each dict should have 'start' and 'end' character offsets.

        Returns
        -------
        list of NegationScope
            All detected negation/context scopes.
        """
        all_scopes = []

        # Pre-negation (forward-scoping)
        all_scopes.extend(
            self._find_triggers(text, self._pre_neg, "forward", NegationStatus.NEGATED)
        )
        # Post-negation (backward-scoping)
        all_scopes.extend(
            self._find_triggers(text, self._post_neg, "backward", NegationStatus.NEGATED)
        )
        # Possibility
        all_scopes.extend(
            self._find_triggers(text, self._possible, "forward", NegationStatus.POSSIBLE)
        )
        # Historical
        all_scopes.extend(
            self._find_triggers(text, self._historical, "forward", NegationStatus.HISTORICAL)
        )
        # Family
        all_scopes.extend(
            self._find_triggers(text, self._family, "forward", NegationStatus.FAMILY)
        )

        if entities is not None:
            # Filter to only scopes overlapping with entities
            relevant = []
            for scope in all_scopes:
                for ent in entities:
                    if scope.scope_start <= ent["start"] and ent["end"] <= scope.scope_end:
                        relevant.append(scope)
                        break
            return relevant

        return all_scopes

    def annotate_entities(
        self,
        text: str,
        entities: List[Dict],
    ) -> List[Dict]:
        """
        Annotate entities with their negation/context status.

        Parameters
        ----------
        text : str
            Clinical text.
        entities : list of dict
            Entities with at least 'start', 'end', 'text', 'label' keys.

        Returns
        -------
        list of dict
            Same entities with added 'negation' key indicating status.
            Priority: NEGATED > POSSIBLE > HISTORICAL > FAMILY > AFFIRMED
        """
        scopes = self.detect(text)
        annotated = []

        for entity in entities:
            ent_start = entity.get("start", entity.get("start_char", 0))
            ent_end = entity.get("end", entity.get("end_char", 0))

            # Find the strongest matching scope
            status = NegationStatus.AFFIRMED
            best_trigger = None

            # Priority ordering (most specific first)
            priority = {
                NegationStatus.NEGATED: 4,
                NegationStatus.FAMILY: 3,
                NegationStatus.HISTORICAL: 2,
                NegationStatus.POSSIBLE: 1,
                NegationStatus.AFFIRMED: 0,
            }

            for scope in scopes:
                if scope.scope_start <= ent_start and ent_end <= scope.scope_end:
                    if priority.get(scope.status, 0) > priority.get(status, 0):
                        status = scope.status
                        best_trigger = scope.trigger_text

            annotated_ent = dict(entity)
            annotated_ent["negation"] = status.value
            if best_trigger:
                annotated_ent["negation_trigger"] = best_trigger
            annotated.append(annotated_ent)

        return annotated

    def is_negated(self, text: str, entity_start: int, entity_end: int) -> bool:
        """Quick check: is the entity at the given span negated?"""
        scopes = self.detect(text)
        for scope in scopes:
            if (scope.status == NegationStatus.NEGATED
                    and scope.scope_start <= entity_start
                    and entity_end <= scope.scope_end):
                return True
        return False
