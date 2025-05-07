include(FetchContent)

# TODO: replace with environment variable and fetch from URL if not set.
set(BOK_SRC_DIR "${CMAKE_SOURCE_DIR}/../bok")

FetchContent_Declare(
    vllm-bok SOURCE_DIR 
    ${BOK_SRC_DIR}
    BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-bok
)