# Adds an example program built from <name>.cpp. All examples go to build/bin/, so they
# are easy to find and run: ./build/bin/<name>
function(add_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE inference_core inference_warnings)
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    )
endfunction()

add_example(tensor_basics)
add_example(ops_basics)
add_example(mlp_forward)
