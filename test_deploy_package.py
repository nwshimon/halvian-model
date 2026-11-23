from pathlib import Path
import runpy


DEPLOY = Path(__file__).parent / "deploy"


def test_deploy_package_contains_early_prep_settings_and_code():
    config = runpy.run_path(str(DEPLOY / "config.py"))

    assert config["PREP_SOFT_THRESHOLD"] == 0.40
    assert config["PREP_EXPEDITE_THRESHOLD"] == 0.60
    assert config["PREP_URGENT_THRESHOLD"] == 0.85
    assert (DEPLOY / "early_prep.py").is_file()
    assert (DEPLOY / "data" / "priors.json").is_file()


if __name__ == "__main__":
    test_deploy_package_contains_early_prep_settings_and_code()
    print("PASS  deploy package includes early-prep settings and dependencies")
