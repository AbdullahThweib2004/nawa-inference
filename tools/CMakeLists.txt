# The `nawa` command-line tool: info / eval / predict.
add_executable(nawa nawa_cli.cpp)
target_link_libraries(nawa PRIVATE nawa_image nawa_server inference_warnings)
# `nawa serve` serves web/ from the source tree by default (override with --web).
target_compile_definitions(nawa PRIVATE NAWA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
set_target_properties(nawa PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# CLI smoke tests: run the real binary through CTest.
if(ENABLE_TESTS)
    set(_model ${PROJECT_SOURCE_DIR}/models/mnist_mlp.nawa)
    set(_images ${PROJECT_SOURCE_DIR}/examples/images)

    add_test(NAME cli.info COMMAND nawa info ${_model})
    set_tests_properties(cli.info PROPERTIES PASS_REGULAR_EXPRESSION "Total parameters: 101,770")

    add_test(NAME cli.predict COMMAND nawa predict ${_model} ${_images}/digit2_inverted.png --show)
    set_tests_properties(cli.predict PROPERTIES PASS_REGULAR_EXPRESSION "Predicted digit: 2")

    # --json: one JSON object and nothing else (python/make_figures.py parses it).
    add_test(NAME cli.predict_json
             COMMAND nawa predict ${_model} ${_images}/digit2_inverted.png --no-preprocess --json)
    set_tests_properties(cli.predict_json PROPERTIES PASS_REGULAR_EXPRESSION
                         "^{\"input28\":\\[[^]]+\\],\"prediction\":7,\"preprocessed\":false,\"probabilities\":\\[[^]]+\\]}\n$")

    # Errors must produce a non-zero exit code.
    add_test(NAME cli.unknown_command COMMAND nawa frobnicate)
    add_test(NAME cli.missing_model COMMAND nawa info ${PROJECT_SOURCE_DIR}/no/such/model.nawa)
    set_tests_properties(cli.unknown_command cli.missing_model PROPERTIES WILL_FAIL TRUE)

    # Web demo smoke test: starts the real `nawa serve` on a free port, calls the JSON API with a
    # fixture image, checks the predicted digit, and stops the server (tests/server_smoke.py).
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME cli.serve_smoke
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/server_smoke.py
                         $<TARGET_FILE:nawa> ${PROJECT_SOURCE_DIR})
        set_tests_properties(cli.serve_smoke PROPERTIES PASS_REGULAR_EXPRESSION "PASS" TIMEOUT 60)
    endif()
endif()
