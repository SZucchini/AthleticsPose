"""Static color definitions."""

mocap_joint_colors = [
    "blue",  # PELVIS
    "blue",  # C_ASIS
    "blue",  # SACR
    "blue",  # C_HIP
    "blue",  # RASI
    "blue",  # RPSI
    "blue",  # RHIP
    "blue",  # RTRO
    "blue",  # RTHI
    "blue",  # RKNE
    "blue",  # RKNM
    "blue",  # RKNL
    "blue",  # R_Medial Tibial Condyle
    "blue",  # R_Lateral Tibial Condyle
    "blue",  # RTIB
    "blue",  # RANK
    "blue",  # RANM
    "blue",  # RANL
    "blue",  # RHEL
    "blue",  # R_Toe
    "blue",  # RBAM
    "blue",  # RMT5
    "red",  # LASI
    "red",  # LPSI
    "red",  # LHIP
    "red",  # LTRO
    "red",  # LTHI
    "red",  # LKNE
    "red",  # LKNM
    "red",  # LKNL
    "red",  # L_Medial Tibial Condyle
    "red",  # L_Lateral Tibial Condyle
    "red",  # LTIB
    "red",  # LANK
    "red",  # LANM
    "red",  # LANL
    "red",  # LHEL
    "red",  # L_Toe
    "red",  # LBAM
    "red",  # LMT5
    "blue",  # Xiphoid
    "blue",  # T10
    "blue",  # RRIB
    "red",  # LRIB
    "blue",  # Manubrium
    "blue",  # CLAV
    "blue",  # C7
    "blue",  # EARC
    "blue",  # Nasion
    "blue",  # BHED
    "blue",  # REAR
    "red",  # LEAR
    "blue",  # Vertex
    "red",  # L_Shoulder Joint
    "red",  # LSHO
    "red",  # LSHF
    "red",  # LSHB
    "red",  # LUPA
    "red",  # LELB
    "red",  # LFRA
    "red",  # LELL
    "red",  # LELM
    "red",  # LWRT
    "red",  # LWSL
    "red",  # LWSM
    "red",  # LHND
    "red",  # LMP2
    "red",  # LMP5
    "blue",  # R_Shoulder Joint
    "blue",  # RSHO
    "blue",  # RSHF
    "blue",  # RSHB
    "blue",  # RUPA
    "blue",  # RELB
    "blue",  # RFRA
    "blue",  # RELL
    "blue",  # RELM
    "blue",  # RWRT
    "blue",  # RWSL
    "blue",  # RWSM
    "blue",  # RHND
    "blue",  # RMP2
    "blue",  # RMP5
    "blue",  # RHIP_Z
]

h36m_joint_colors = [
    "blue",
    "blue",
    "blue",
    "blue",
    "red",
    "red",
    "red",
    "blue",
    "blue",
    "blue",
    "blue",
    "red",
    "red",
    "red",
    "blue",
    "blue",
    "blue",
]

mocap_bone_colors = [
    "blue",  # PELVIS - C_HIP
    "blue",  # PELVIS - CLAV
    "blue",  # C_ASIS - RASI
    "red",  # C_ASIS - LASI
    "blue",  # SACR - RPSI
    "red",  # SACR - LPSI
    "blue",  # C_HIP - RHIP
    "red",  # C_HIP - LHIP
    "blue",  # RASI - RHIP
    "blue",  # RASI - RTRO
    "blue",  # RASI - RTHI
    "blue",  # RASI - RHIP_Z
    "blue",  # RPSI - RHIP
    "blue",  # RPSI - RTRO
    "blue",  # RPSI - RHIP_Z
    "blue",  # RHIP - RTHI
    "blue",  # RTRO - RTHI
    "blue",  # RTHI - RKNE
    "blue",  # RKNE - RKNM
    "blue",  # RKNE - RKNL
    "blue",  # RKNE - RTIB
    "blue",  # RKNM - R_Medial Tibial Condyle
    "blue",  # RKNL - R_Lateral Tibial Condyle
    "blue",  # R_Medial Tibial Condyle - R_Lateral Tibial Condyle
    "blue",  # RTIB - RANK
    "blue",  # RANK - RANM
    "blue",  # RANK - RANL
    "blue",  # RANK - RHEL
    "blue",  # RANK - R_Toe
    "blue",  # RANM - RBAM
    "blue",  # RANL - RMT5
    "red",  # LASI - LPSI
    "red",  # LASI - LHIP
    "red",  # LASI - LTRO
    "red",  # LPSI - LHIP
    "red",  # LPSI - LTRO
    "red",  # LHIP - LTHI
    "red",  # LTRO - LTHI
    "red",  # LTHI - LKNE
    "red",  # LKNE - LKNM
    "red",  # LKNE - LKNL
    "red",  # LKNE - LTIB
    "red",  # LKNM - L_Medial Tibial Condyle
    "red",  # LKNL - L_Lateral Tibial Condyle
    "red",  # L_Medial Tibial Condyle - L_Lateral Tibial Condyle
    "red",  # LTIB - LANK
    "red",  # LANK - LANM
    "red",  # LANK - LANL
    "red",  # LANK - LHEL
    "red",  # LANK - L_Toe
    "red",  # LANM - LBAM
    "red",  # LANL - LMT5
    "blue",  # Xiphoid - RRIB
    "red",  # Xiphoid - LRIB
    "blue",  # T10 - RRIB
    "red",  # T10 - LRIB
    "red",  # Manubrium - LSHF
    "blue",  # Manubrium - RSHF
    "blue",  # CLAV - EARC
    "red",  # CLAV - LSHO
    "blue",  # CLAV - RSHO
    "red",  # C7 - LSHB
    "blue",  # C7 - RSHB
    "blue",  # EARC - Vertex
    "blue",  # Nasion - REAR
    "red",  # Nasion - LEAR
    "blue",  # BHED - REAR
    "red",  # BHED - LEAR
    "red",  # L_Shoulder Joint - LSHF
    "red",  # L_Shoulder Joint - LSHB
    "red",  # L_Shoulder Joint - LUPA
    "red",  # L_Shoulder Joint - LELB
    "red",  # LSHO - LSHF
    "red",  # LSHO - LSHB
    "red",  # LELB - LELL
    "red",  # LELB - LELM
    "red",  # LELB - LWRT
    "red",  # LFRA - LELL
    "red",  # LFRA - LELM
    "red",  # LWRT - LWSL
    "red",  # LWRT - LWSM
    "red",  # LWRT - LHND
    "red",  # LWSL - LMP2
    "red",  # LWSM - LMP5
    "blue",  # R_Shoulder Joint - RSHF
    "blue",  # R_Shoulder Joint - RSHB
    "blue",  # R_Shoulder Joint - RUPA
    "blue",  # RSHO - RSHF
    "blue",  # RSHO - RSHB
    "blue",  # RUPA - RELB
    "blue",  # RELB - RELL
    "blue",  # RELB - RELM
    "blue",  # RELB - RWRT
    "blue",  # RFRA - RELL
    "blue",  # RFRA - RELM
    "blue",  # RWRT - RWSL
    "blue",  # RWRT - RWSM
    "blue",  # RWRT - RHND
    "blue",  # RWSL - RMP2
    "blue",  # RWSM - RMP5
]

h36m_bone_colors = [
    "blue",
    "red",
    "blue",
    "red",
    "blue",
    "red",
    "blue",
    "blue",
    "blue",
    "blue",
    "blue",
    "red",
    "blue",
    "red",
    "blue",
    "red",
]
