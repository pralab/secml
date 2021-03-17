from .c_attack_evasion import CAttackEvasion
from .c_attack_evasion_pgd_ls import CAttackEvasionPGDLS
from .c_attack_evasion_pgd_exp import CAttackEvasionPGDExp
from .c_attack_evasion_pgd import CAttackEvasionPGD

try:
    import cleverhans
except ImportError:
    pass  # cleverhans is an extra component
else:
    from .cleverhans.c_attack_evasion_cleverhans import \
        CAttackEvasionCleverhans

try:
    import foolbox
    import torch
except ImportError:
    pass  # foolbox is an extra component and requires pytorch
else:
    from .foolbox.secml_autograd import SecmlLayer
    from .foolbox.c_attack_evasion_foolbox import \
        CAttackEvasionFoolbox
    from .foolbox.fb_attacks.fb_basic_iterative_attack import \
        CFoolboxBasicIterative, CFoolboxBasicIterativeL1, \
        CFoolboxBasicIterativeL2, CFoolboxBasicIterativeLinf
    from .foolbox.fb_attacks.fb_cw_attack import \
        CFoolboxL2CarliniWagner
    from .foolbox.fb_attacks.fb_ddn_attack import CFoolboxL2DDN
    from .foolbox.fb_attacks.fb_deepfool_attack import \
        CFoolboxDeepfool, CFoolboxDeepfoolL2, CFoolboxDeepfoolLinf
    from .foolbox.fb_attacks.fb_ead_attack import CFoolboxEAD
    from .foolbox.fb_attacks.fb_fgm_attack import \
        CFoolboxFGM, CFoolboxFGML1, CFoolboxFGML2, CFoolboxFGMLinf
    from .foolbox.fb_attacks.fb_pgd_attack import \
        CFoolboxPGD, CFoolboxPGDL1, CFoolboxPGDL2, CFoolboxPGDLinf
