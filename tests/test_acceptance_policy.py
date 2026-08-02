from math_agent_core.acceptance import AcceptancePolicy
from math_agent_core.state import EvidenceStatus, VerificationEvidence, VerificationLevel


def _evidence(status, level, decisive, method="check", verifier="tool"):
    return VerificationEvidence(
        verifier=verifier,
        claim_id="c",
        status=status,
        method=method,
        details="details",
        verification_level=level,
        is_decisive=decisive,
    )


def test_exact_symbolic_pass_accepts_solved():
    decision = AcceptancePolicy().decide(
        schema_valid=True,
        content_complete=True,
        task_type="calculation",
        answer_type="numeric",
        model_verification_pass=False,
        evidence=[_evidence(EvidenceStatus.PASS.value, VerificationLevel.EXACT_SYMBOLIC.value, True)],
    )

    assert decision.overall_status == "solved"
    assert decision.answer_verified is True


def test_decisive_tool_fail_rejects_even_when_model_passes():
    decision = AcceptancePolicy().decide(
        schema_valid=True,
        content_complete=True,
        task_type="calculation",
        answer_type="numeric",
        model_verification_pass=True,
        evidence=[
            _evidence(EvidenceStatus.FAIL.value, VerificationLevel.EXACT_SYMBOLIC.value, True, method="equation_solution"),
            _evidence(EvidenceStatus.PASS.value, VerificationLevel.MODEL_CRITIC.value, False, verifier="critic"),
        ],
    )

    assert decision.overall_status == "invalid"
    assert decision.answer_verified is False


def test_high_precision_numeric_is_probable_not_solved():
    decision = AcceptancePolicy().decide(
        schema_valid=True,
        content_complete=True,
        task_type="calculation",
        answer_type="numeric",
        model_verification_pass=False,
        evidence=[_evidence(EvidenceStatus.PASS.value, VerificationLevel.HIGH_PRECISION_NUMERIC.value, False)],
    )

    assert decision.overall_status == "probable"
    assert decision.answer_verified is False


def test_completeness_only_pass_does_not_accept_math():
    decision = AcceptancePolicy().decide(
        schema_valid=True,
        content_complete=True,
        task_type="calculation",
        answer_type="text",
        model_verification_pass=False,
        evidence=[_evidence(EvidenceStatus.PASS.value, VerificationLevel.COMPLETENESS_ONLY.value, False, verifier="completeness")],
    )

    assert decision.overall_status == "uncertain"


def test_model_critic_only_is_probable_not_solved():
    decision = AcceptancePolicy().decide(
        schema_valid=True,
        content_complete=True,
        task_type="calculation",
        answer_type="text",
        model_verification_pass=False,
        evidence=[_evidence(EvidenceStatus.PASS.value, VerificationLevel.MODEL_CRITIC.value, False, verifier="critic")],
    )

    assert decision.overall_status == "probable"
    assert decision.answer_verified is False
