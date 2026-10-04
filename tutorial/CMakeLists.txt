# Tutorials (tutorial/*/README.md). Each is one self-contained program; nothing else in strixite depends on them.
add_executable(first_kernel)
strix_hip_sources(first_kernel first-kernel/first_kernel.hip)
target_link_libraries(first_kernel PRIVATE strix_rocm)
