# The `nawa` command-line tool: info / eval / predict.
add_executable(nawa nawa_cli.cpp)
target_link_libraries(nawa PRIVATE nawa_image inference_warnings)
set_target_properties(nawa PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# CLI smoke tests: run the real binary through CTest.
if(ENABLE_TESTS)
    set(_model ${PROJECT_SOURCE_DIR}/models/mnist_mlp.nawa)
    set(_images ${PROJECT_SOURCE_DIR}/examples/images)

    add_test(NAME cli.info COMMAND nawa info ${_model})
    set_tests_properties(cli.info PROPERTIES PASS_REGULAR_EXPRESSION "Total parameters: 101,770")

    add_test(NAME cli.predict COMMAND nawa predict ${_model} ${_images}/digit2_inverted.png --show)
    set_tests_properties(cli.predict PROPERTIES PASS_REGULAR_EXPRESSION "Predicted digit: 2")

    # Errors must produce a non-zero exit code.
    add_test(NAME cli.unknown_command COMMAND nawa frobnicate)
    add_test(NAME cli.missing_model COMMAND nawa info ${PROJECT_SOURCE_DIR}/no/such/model.nawa)
    set_tests_properties(cli.unknown_command cli.missing_model PROPERTIES WILL_FAIL TRUE)
endif()
