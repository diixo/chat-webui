"""
Test intent classifier for automotive 2-wheel drive vehicle sensor domain.
Covers ASPICE-relevant tracker types.

Run:
    python test_classifier.py
"""

from utils import load_model, classify_intent

from codebeamer_app import SYSTEM_PROMPT


TEST_CASES = [
    # (user_input, expected_tracker_type)

    # --- Requirement ---
    ("create requirement: Battery voltage sensor shall measure range 9V to 16V",            "Requirement"),
    ("create requirement: ABS wheel speed sensor shall sample at minimum 1kHz",             "Requirement"),
    ("create requirement: Temperature sensor shall report values in Celsius",               "Requirement"),
    ("create requirement: Brake fluid pressure sensor accuracy ±0.5%",                     "Requirement"),
    ("create requirement: Motor phase current sensor shall detect overcurrent within 5ms",  "Requirement"),

    # --- Bug ---
    ("create bug: ABS sensor returns zero speed at vehicle start",                          "Bug"),
    ("create bug: Battery temperature sensor reads -40°C when engine is warm",             "Bug"),
    ("create bug: Wheel speed sensor signal lost intermittently above 60 km/h",            "Bug"),
    ("create bug: Throttle position sensor calibration drifts after firmware update",      "Bug"),

    # --- Task ---
    ("create task: Integrate CAN driver for ABS sensor data transmission",                 "Task"),
    ("create task: Calibrate battery state-of-charge sensor for low temperature",          "Task"),
    ("create task: Implement diagnostic routine for motor temperature sensor",              "Task"),
    ("create task: Write unit tests for wheel speed signal filtering algorithm",           "Task"),

    # --- Epic ---
    ("create epic: Sensor Diagnostics Framework for 2-wheel drive platform",               "Epic"),
    ("create epic: Battery Management Sensor Integration",                                  "Epic"),
    ("create epic: ABS Sensor Signal Processing Pipeline",                                  "Epic"),

    # --- User Story ---
    ("create user story: As a driver I want the dashboard to warn me when battery temperature exceeds 45°C", "User Story"),
    ("create user story: As a technician I want to read raw ABS sensor data via OBD port",  "User Story"),
    ("create user story: As a system I want wheel speed sensors to trigger ABS within 20ms of slip detection", "User Story"),

    # --- Risk ---
    ("create risk: ABS sensor contamination by road debris may cause false slip detection", "Risk"),
    ("create risk: Battery temperature sensor failure may lead to undetected thermal runaway", "Risk"),
    ("create risk: Single-point failure in wheel speed sensor wiring harness",             "Risk"),

    # --- Test Case ---
    ("create test case: Verify ABS sensor output at 0 km/h and 120 km/h",                 "Test Case"),
    ("create test case: Battery voltage sensor accuracy under load discharge cycle",        "Test Case"),
    ("create test case: Temperature sensor response time when coolant exceeds 90°C",       "Test Case"),
    ("create test case: Motor current sensor overcurrent detection latency test",           "Test Case"),

    # --- Test Set ---
    ("create test set: ABS sensor functional tests for 2-wheel drive",                     "Test Set"),
    ("create test set: Battery sensor regression suite",                                   "Test Set"),

    # --- Test Run ---
    ("create test run: ABS sensor suite on firmware v2.3.1",                               "Test Run"),
    ("create test run: Battery temperature sensor tests on hardware revision B",           "Test Run"),

    # --- Test Configuration ---
    ("create test configuration: HIL bench with real ABS wheel speed sensors",             "Test Configuration"),
    ("create test configuration: SIL environment for battery voltage sensor simulation",   "Test Configuration"),

    # --- Issue ---
    ("create issue: ABS sensor CAN message missing during cold start",                     "Issue"),
    ("create issue: Sporadic CRC errors on battery sensor CAN frame",                      "Issue"),

    # --- Change Request ---
    ("create change request: Increase battery temperature sensor sampling rate from 10Hz to 50Hz", "Change Request"),
    ("create change request: Add redundant wheel speed sensor input to ABS ECU",           "Change Request"),

    # --- Release ---
    ("create release: Sensor Diagnostics Module v1.0 for 2WD platform",                   "Release"),
    ("create release: ABS sensor firmware patch release 2.3.2",                            "Release"),

    # --- Component ---
    ("create component: Wheel Speed Sensor Driver",                                         "Component"),
    ("create component: Battery Voltage Monitor Module",                                    "Component"),
    ("create component: ABS Control Unit sensor interface",                                 "Component"),

    # --- should NOT match (unknown) ---
    ("did you create the ABS sensor requirement?",                                         None),
    ("was the battery sensor test case created?",                                          None),
    ("how are you?",                                                                       None),
    ("have you already created the risk?",                                                 None),
]


def run_tests():
    print("Loading model...")
    tokenizer, model = load_model("models/Qwen3-0.6B")
    print(f"Model loaded. Running {len(TEST_CASES)} test cases...\n")

    passed = 0
    failed = 0

    for user_input, expected_type in TEST_CASES:
        result = classify_intent(tokenizer, model, user_input, SYSTEM_PROMPT)
        intent = result["intent"]
        tracker_type = result["tracker_type"]
        description = result["description"]

        if expected_type is None:
            ok = intent == "unknown"
        else:
            ok = intent == "create_tracker" and tracker_type == expected_type

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {user_input[:70]}")
        if not ok:
            if expected_type is None:
                print(f"       expected: unknown")
            else:
                print(f"       expected: create_tracker / {expected_type}")
            print(f"       got:      {intent} / {tracker_type!r}")
            print(f"       description: {description!r}")
        print()

    total = passed + failed
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
    else:
        print("  — all passed!")


if __name__ == "__main__":
    run_tests()
