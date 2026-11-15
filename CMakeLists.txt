cmake_minimum_required(VERSION 3.16)
project(igabem_elasticity CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)
# UMFPACK (SuiteSparse): sparse LU of the hierarchical solvers' preconditioner (src/hmat), used by BcModel.
find_library(UMFPACK_LIB umfpack REQUIRED)
find_path(UMFPACK_INC umfpack.h PATH_SUFFIXES suitesparse REQUIRED)

foreach(app patch_test dirichlet_test interp_error cavity_test plate_test cad_test bc_case bc_validate region_validate rve_cell bead_cell crack_case crack_prism crack_serve crack_tension crack_grow)
  add_executable(${app} app/${app}.cpp)
  target_include_directories(${app} PRIVATE src)
  target_link_libraries(${app} PRIVATE Eigen3::Eigen)
  target_compile_options(${app} PRIVATE -O3 -march=native -Wall -Wextra)
  target_compile_definitions(${app} PRIVATE IGABEM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  target_link_libraries(${app} PRIVATE ${UMFPACK_LIB})
  target_include_directories(${app} SYSTEM PRIVATE ${UMFPACK_INC})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${app} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()

execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE IGABEM_GIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(IGABEM_GIT)
  target_compile_definitions(crack_serve PRIVATE IGABEM_GIT="${IGABEM_GIT}")
endif()

enable_testing()
add_executable(test_kernels tests/test_kernels.cpp)
target_include_directories(test_kernels PRIVATE src)
target_link_libraries(test_kernels PRIVATE Eigen3::Eigen)
add_test(NAME kernels COMMAND test_kernels)
add_executable(test_regression tests/test_regression.cpp)
target_include_directories(test_regression PRIVATE src)
target_link_libraries(test_regression PRIVATE Eigen3::Eigen)
target_compile_options(test_regression PRIVATE -O3 -march=native)
if(OpenMP_CXX_FOUND)
  target_link_libraries(test_regression PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME regression COMMAND test_regression ${CMAKE_SOURCE_DIR})
# XIGABEM crack kernels (src/crack) and the penny regression (paper Sec. 4.1, O-grid n = 5, K_I within 1%)
add_executable(test_crack tests/test_crack.cpp)
target_include_directories(test_crack PRIVATE src)
target_link_libraries(test_crack PRIVATE Eigen3::Eigen)
target_compile_options(test_crack PRIVATE -O3 -march=native)
target_compile_definitions(test_crack PRIVATE IGABEM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME crack_kernels COMMAND test_crack)
add_test(NAME crack_penny COMMAND crack_case ${CMAKE_SOURCE_DIR}/geo/crack/penny_og_i0_n5.dat --check-k 0.01)
# end-shift independence: inner patch graded to the crown's element size (make_crack --inner-grade 1.5); n7 K_I error
# -3.5e-5 / 2.0e-5 / 6.3e-5 at shift 0.02 / 0.1 / 0.2 (uniform inner patch: -7e-5 / 4.6e-4 / 9.7e-4)
foreach(sh 0.02 0.2)
  add_test(NAME crack_penny_shift_${sh} COMMAND crack_case ${CMAKE_SOURCE_DIR}/geo/crack/penny_og_i0_n7_ig.dat
           --shift ${sh} --check-k 1e-4)
endforeach()
# ellipse a = 2b on the ring layout (no degenerate corner; crown within half the front's smallest radius of
# curvature, inner patches graded): n7 signed K_I error 1.9e-4 / 2.2e-4 / 2.0e-4 at shift 0.02 / 0.1 / 0.2, mean
# |error| along the front 4.7e-4
foreach(sh 0.02 0.2)
  add_test(NAME crack_ellipse_shift_${sh} COMMAND crack_case ${CMAKE_SOURCE_DIR}/geo/crack/ellipse_ring_i0_n7_ig.dat
           --nu 0.3 --load pressure --shift ${sh} --check-k 6e-4)
endforeach()
# central-patch corner rows (coincident-point search): cond was 2.3e12 and the LU residual 3e-7 before the fix
add_test(NAME crack_penny_conditioning COMMAND crack_case ${CMAKE_SOURCE_DIR}/geo/crack/penny_i0_n4.dat --check-cond 1e7 --check-k 1.2e-2)
# independent K: interaction integral on the interior field (crack/interaction.hpp); n4 measures 5.8e-5
add_test(NAME crack_penny_mint COMMAND crack_case ${CMAKE_SOURCE_DIR}/geo/crack/penny_i0_n4.dat --mint 0.1 --check-mint 1.2e-2)
add_test(NAME crack_grow_reuse COMMAND crack_grow ${CMAKE_SOURCE_DIR}/geo/crack/senb_grow_aw0.5_n2_s0.dat ${CMAKE_SOURCE_DIR}/geo/crack/senb_grow_aw0.5_n2_s1.dat --verify --check-k 1e-8)
add_test(NAME crack_penny_inclined COMMAND crack_case ${CMAKE_SOURCE_DIR}/geo/crack/penny_og_i45_n5.dat --check-k 0.01)
add_test(NAME crack_ellipse COMMAND crack_case ${CMAKE_SOURCE_DIR}/geo/crack/ellipse_i0_n5.dat --nu 0.3 --load pressure --check-k 0.02)
add_test(NAME crack_prism_tension COMMAND crack_tension ${CMAKE_SOURCE_DIR}/geo/crack/prismt_split_c1.dat --check 0.02)
add_test(NAME crack_prism COMMAND crack_prism ${CMAKE_SOURCE_DIR}/geo/crack/prism_k3.dat --check-k 2e-4)
# r^3/2 remainder + fixed-zone front grading (convergence review): 9.4% at shift 0.1 ungraded, 3.4e-3 here
# r^3/2 Williams term: the prism with the a3 term, uniform mesh (first term only: 9.4e-2)
add_test(NAME crack_prism_a3_second COMMAND crack_prism ${CMAKE_SOURCE_DIR}/geo/crack/prism_split_k3.dat --a3 1 --check-k 2e-4)
add_test(NAME crack_prism_a3_graded COMMAND crack_prism ${CMAKE_SOURCE_DIR}/geo/crack/prism_split_k3_g.dat --a3 1 --check-k 5e-3)
# the factored path crack_serve uses (factor once, right-hand sides alone), and the server's JSON round trip
add_test(NAME crack_penny_inclined_factored COMMAND crack_case ${CMAKE_SOURCE_DIR}/geo/crack/penny_og_i45_n5.dat --check-k 0.01 --factored)
add_test(NAME crack_prism_factored COMMAND crack_prism ${CMAKE_SOURCE_DIR}/geo/crack/prism_k3.dat --check-k 2e-4 --factored)
add_test(NAME crack_serve_case COMMAND crack_serve --case ${CMAKE_SOURCE_DIR}/tests/crack_serve_penny.json WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
# local supports and loads (BcModel): exact bar fields, half-space disk load, cantilever
add_test(NAME bc_bar COMMAND bc_validate bar ${CMAKE_SOURCE_DIR}/geo/bar_L4_p3_n8.dat --check)
add_test(NAME bc_halfspace COMMAND bc_validate halfspace ${CMAKE_SOURCE_DIR}/geo/box_hs_q_p3_r0.dat --cache none --check)
add_test(NAME bc_cantilever COMMAND bc_validate cantilever ${CMAKE_SOURCE_DIR}/geo/bar_L10_p3_n20.dat --check)
add_test(NAME bc_rbe2_treecode COMMAND bc_validate rbe2 ${CMAKE_SOURCE_DIR}/geo/bar_L10_p3_n20.dat --check)
add_test(NAME bc_plate_knots COMMAND bc_case ${CMAKE_SOURCE_DIR}/cases/plate_clamped_knots.json --max-residual 0.01)
# two-region bodies (MultiRegion): exact patch / Lame / Eshelby fields, printed coupon
add_test(NAME region_patch COMMAND region_validate patch ${CMAKE_SOURCE_DIR}/geo/regions/sph2_a0.5_r1_host.dat ${CMAKE_SOURCE_DIR}/geo/regions/sph2_a0.5_r1_core.dat --check)
add_test(NAME region_lame COMMAND region_validate lame ${CMAKE_SOURCE_DIR}/geo/regions/sph2_a0.5_r1_host.dat ${CMAKE_SOURCE_DIR}/geo/regions/sph2_a0.5_r1_core.dat --check)
add_test(NAME region_eshelby COMMAND region_validate eshelby ${CMAKE_SOURCE_DIR}/geo/regions/sph2_a0.5_r1_host.dat ${CMAKE_SOURCE_DIR}/geo/regions/sph2_a0.5_r1_core.dat --check)
add_test(NAME rve_solid COMMAND rve_cell ${CMAKE_SOURCE_DIR}/geo/cube_p2.dat --bc kubc,pbc,subc --ng 16 --check-iso 1e-6)
add_test(NAME rve_sphere_pore COMMAND rve_cell ${CMAKE_SOURCE_DIR}/geo/cube_cavity_r1.dat --bc pbc --check-c 3246.21,3246.21,761.765,761.765 --tol 1e-5)
add_test(NAME region_coupon COMMAND region_validate coupon ${CMAKE_SOURCE_DIR}/geo/regions/coupon_w0.15_n2_host.dat ${CMAKE_SOURCE_DIR}/geo/regions/coupon_w0.15_n2_core.dat --rho 0,0.2,1 --check)
# the same on the hierarchical backend (src/hmat/h_mixed.hpp) at eps 1e-8, to the float storage floor
foreach(c bar:bar_L4_p3_n8 halfspace:box_hs_q_p3_r0 cantilever:bar_L10_p3_n20 rbe2:bar_L10_p3_n20)
  string(REPLACE ":" ";" cg ${c})
  list(GET cg 0 cn)
  list(GET cg 1 cgeo)
  add_test(NAME bc_${cn}_hmat COMMAND bc_validate ${cn} ${CMAKE_SOURCE_DIR}/geo/${cgeo}.dat --cache none --check --hmat 1e-8)
endforeach()
foreach(c patch lame eshelby)
  add_test(NAME region_${c}_hmat COMMAND region_validate ${c} ${CMAKE_SOURCE_DIR}/geo/regions/sph2_a0.5_r1_host.dat ${CMAKE_SOURCE_DIR}/geo/regions/sph2_a0.5_r1_core.dat --check --hmat 1e-8)
endforeach()
add_test(NAME region_coupon_hmat COMMAND region_validate coupon ${CMAKE_SOURCE_DIR}/geo/regions/coupon_w0.15_n2_host.dat ${CMAKE_SOURCE_DIR}/geo/regions/coupon_w0.15_n2_core.dat --rho 0,0.2,1 --check --hmat 1e-8)
add_test(NAME bc_bracket_bolted_hmat COMMAND bc_case ${CMAKE_SOURCE_DIR}/cases/bracket_bolted_hmat.json --max-residual 0.01)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  add_test(NAME failure_layer_unit COMMAND ${Python3_EXECUTABLE} -m unittest failure.test_failure
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
  add_test(NAME failure_orientation_unit COMMAND ${Python3_EXECUTABLE} -m unittest failure.test_aniso
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
  add_test(NAME model_tree_unit COMMAND ${Python3_EXECUTABLE} -m unittest igamodel.test_igamodel
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
  add_test(NAME model_backends_unit COMMAND ${Python3_EXECUTABLE} -m unittest igamodel.test_backends
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
  add_test(NAME model_bc_case COMMAND ${Python3_EXECUTABLE} -m unittest igamodel.test_bc_case
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
  set_tests_properties(model_bc_case PROPERTIES ENVIRONMENT IGABEM_BC_CASE=$<TARGET_FILE:bc_case>)
  add_test(NAME rve_unit COMMAND ${Python3_EXECUTABLE} -m unittest rve.test_rve
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
  set_tests_properties(rve_unit PROPERTIES ENVIRONMENT "IGABEM_RVE_CELL=$<TARGET_FILE:rve_cell>;IGABEM_BEAD_CELL=$<TARGET_FILE:bead_cell>")
  # fatigue transfer harness (python/fatigue): criteria and dataset checks, no solve, < 1 s
  add_test(NAME fatigue_unit COMMAND ${Python3_EXECUTABLE} -m pytest -q fatigue/test_fatigue.py
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
  # SENB 3-point bend K_I (senb_k.py): two-mesh extrapolated K vs ASTM D5045 within 2 %, nu 0, ~10 s
  add_test(NAME crack_senb_d5045 COMMAND ${Python3_EXECUTABLE} senb_k.py --check
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
  set_tests_properties(crack_senb_d5045 PROPERTIES ENVIRONMENT IGABEM_CRACK_SERVE=$<TARGET_FILE:crack_serve>)
endif()

# Hierarchical compression (src/fast): Htool (third_party/htool, MIT, header-only) + BLAS/LAPACK.
find_package(BLAS)
find_package(LAPACK)
if(BLAS_FOUND AND LAPACK_FOUND)
  foreach(t test_hmatrix test_dense_fast hmatrix_bench near_profile dense_bench hmat_bench)
    if(t MATCHES "^test_")
      add_executable(${t} tests/${t}.cpp)
    else()
      add_executable(${t} app/${t}.cpp)
    endif()
    target_include_directories(${t} PRIVATE src)
    target_include_directories(${t} SYSTEM PRIVATE third_party/htool/include)
    target_link_libraries(${t} PRIVATE Eigen3::Eigen ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    target_compile_options(${t} PRIVATE -O3 -march=native)
    target_link_libraries(${t} PRIVATE ${UMFPACK_LIB})
    target_include_directories(${t} SYSTEM PRIVATE ${UMFPACK_INC})
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${t} PRIVATE OpenMP::OpenMP_CXX)
    endif()
  endforeach()
  # Bead cells: LAPACK LU for the condensation and the weld system (src/bead_cell.hpp DenseLU)
  target_link_libraries(bead_cell PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  target_compile_definitions(bead_cell PRIVATE IGABEM_LAPACK)
  add_test(NAME hmatrix COMMAND test_hmatrix ${CMAKE_SOURCE_DIR})
  set_tests_properties(hmatrix PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
  add_test(NAME dense_fast COMMAND test_dense_fast ${CMAKE_SOURCE_DIR})
  # 5.8k plate: H x and G c against DenseSbie (<= 10 eps), Kirsch error at the dense value (3.846e-5)
  add_test(NAME hmat_kirsch COMMAND hmat_bench ${CMAKE_SOURCE_DIR}/geo/plate_hole_p4_t0.2_r3.dat --check --max-error 3.9e-5)
  set_tests_properties(hmat_kirsch PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
else()
  message(STATUS "BLAS/LAPACK not found: hierarchical compression (src/fast) not built")
endif()
