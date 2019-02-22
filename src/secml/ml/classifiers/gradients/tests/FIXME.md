This entire tests unit needs revision.

1. Create one TestCase for each CClassifierGradient class
2. In each TestCase, create one entrypoint for each test unit, es. test_L_params_gradient
3. Move "static" routines to setUpClass (es. DS creation)
4. Remove unnecessary/redundant integration tests