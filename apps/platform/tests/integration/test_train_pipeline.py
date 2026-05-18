from odp_platform.training.service import TrainingService


def test_training_service_placeholder_status() -> None:
    service = TrainingService()
    assert service.status() == "not-implemented"
