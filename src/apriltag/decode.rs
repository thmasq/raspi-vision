use crate::apriltag::image::Image;
use crate::apriltag::pose::{CameraIntrinsics, estimate_tag_pose};
use crate::apriltag::quad::refine_edges;
use nalgebra::{SMatrix, SVector};
use serde::Serialize;

pub const TAG36H11_WIDTH_AT_BORDER: usize = 8;
#[allow(dead_code)]
pub const TAG36H11_TOTAL_WIDTH: usize = 10;
pub const TAG36H11_NBITS: usize = 36;
pub const TAG36H11_MAX_HAMMING: u32 = 2;

pub const TAG36H11_BIT_X: [usize; 36] = [
    1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4, 1, 1, 1, 1, 1,
    2, 2, 2, 3,
];

pub const TAG36H11_BIT_Y: [usize; 36] = [
    1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2,
    5, 4, 3, 4,
];

pub const TAG36H11_CODES: [u64; 587] = [
    0x0000_000d_7e00_984b,
    0x0000_000d_da66_4ca7,
    0x0000_000d_c4a1_c821,
    0x0000_000e_17b4_70e9,
    0x0000_000e_f91d_01b1,
    0x0000_000f_429c_dd73,
    0x0000_0000_5da2_9225,
    0x0000_0001_106c_ba43,
    0x0000_0002_23be_d79d,
    0x0000_0002_1f51_213c,
    0x0000_0003_3eb1_9ca6,
    0x0000_0003_f76e_b0f8,
    0x0000_0004_69a9_7414,
    0x0000_0004_5dcf_e0b0,
    0x0000_0004_a646_5f72,
    0x0000_0005_1801_db96,
    0x0000_0005_eb94_6b4e,
    0x0000_0006_8a7c_c2ec,
    0x0000_0006_f0ba_2652,
    0x0000_0007_8765_559d,
    0x0000_0008_7b83_d129,
    0x0000_0008_6cc4_a5c5,
    0x0000_0008_b64d_f90f,
    0x0000_0009_c577_b611,
    0x0000_000a_3810_f2f5,
    0x0000_000a_f4d7_5b83,
    0x0000_000b_59a0_3fef,
    0x0000_000b_b109_6f85,
    0x0000_000d_1b92_fc76,
    0x0000_000d_0dd5_09d2,
    0x0000_000e_2cfd_a160,
    0x0000_0002_ff49_7c63,
    0x0000_0004_7240_671b,
    0x0000_0005_047a_2e55,
    0x0000_0006_35ca_87c7,
    0x0000_0006_9125_4166,
    0x0000_0006_8f43_d94a,
    0x0000_0006_ef24_bdb6,
    0x0000_0008_cdd8_f886,
    0x0000_0009_de96_b718,
    0x0000_000a_ff6e_5a8a,
    0x0000_000b_ae46_f029,
    0x0000_000d_225b_6d59,
    0x0000_000d_f8ba_8c01,
    0x0000_000e_3744_a22f,
    0x0000_000f_bb59_375d,
    0x0000_0001_8a91_6828,
    0x0000_0002_2f29_c1ba,
    0x0000_0002_8688_7d58,
    0x0000_0004_1392_322e,
    0x0000_0007_5d18_ecd1,
    0x0000_0008_7c30_2743,
    0x0000_0008_c631_7ba9,
    0x0000_0009_e40f_36d7,
    0x0000_000c_0e5a_806a,
    0x0000_000c_c78c_b87c,
    0x0000_0001_2d2f_2d01,
    0x0000_0003_79f3_6a21,
    0x0000_0006_973f_59ac,
    0x0000_0007_789e_a9f4,
    0x0000_0008_f1c7_3e84,
    0x0000_0008_dd28_7a20,
    0x0000_0009_4a4e_ee4c,
    0x0000_000a_4553_79b5,
    0x0000_000a_9e92_987d,
    0x0000_000b_d25c_b40b,
    0x0000_000b_e98d_3582,
    0x0000_000d_3d59_72b2,
    0x0000_0001_4c53_d7c7,
    0x0000_0004_f179_6936,
    0x0000_0004_e71f_ed1a,
    0x0000_0006_6d46_fae0,
    0x0000_000a_55ab_b933,
    0x0000_000e_bee1_acca,
    0x0000_0001_ad4b_a6a4,
    0x0000_0003_05b1_7571,
    0x0000_0005_5361_1351,
    0x0000_0005_9ca6_2775,
    0x0000_0007_819c_b6a1,
    0x0000_000e_db7b_c9eb,
    0x0000_0005_b269_4212,
    0x0000_0007_2e12_d185,
    0x0000_000e_d615_2e2c,
    0x0000_0005_bcda_dbf3,
    0x0000_0007_8e0a_a0c6,
    0x0000_000c_60a0_b909,
    0x0000_000e_f9a3_4b0d,
    0x0000_0003_98a6_621a,
    0x0000_000a_8a27_c944,
    0x0000_0004_b564_304e,
    0x0000_0005_2902_b4e2,
    0x0000_0008_5728_0b56,
    0x0000_000a_91b2_c84b,
    0x0000_000e_91df_939b,
    0x0000_0001_fa40_5f28,
    0x0000_0002_3793_ab86,
    0x0000_0006_8c17_729f,
    0x0000_0009_fbf3_b840,
    0x0000_0003_6922_413c,
    0x0000_0004_eb5f_946e,
    0x0000_0005_33fe_2404,
    0x0000_0006_3de7_d35e,
    0x0000_0009_25ed_dc72,
    0x0000_0009_9b8b_3896,
    0x0000_000a_ace4_c708,
    0x0000_000c_2299_4af0,
    0x0000_0008_f1ea_e41b,
    0x0000_000d_95fb_486c,
    0x0000_0001_3fb7_7857,
    0x0000_0004_fe09_83a3,
    0x0000_000d_559b_f8a9,
    0x0000_000e_1855_d78d,
    0x0000_000f_ec8d_aaad,
    0x0000_0007_1ecb_6d95,
    0x0000_000d_c9e5_0e4c,
    0x0000_000c_a3a4_c259,
    0x0000_0007_40d1_2bbf,
    0x0000_000a_eedd_18e0,
    0x0000_000b_509b_9c8e,
    0x0000_0005_232f_ea1c,
    0x0000_0001_9282_d18b,
    0x0000_0007_6c22_d67b,
    0x0000_0009_36be_b34b,
    0x0000_0000_8a5e_a8dd,
    0x0000_0006_79ea_dc28,
    0x0000_000a_08e1_19c5,
    0x0000_0002_0a6e_3e24,
    0x0000_0007_eab9_c239,
    0x0000_0009_6632_c32e,
    0x0000_0004_70d0_6e44,
    0x0000_0008_a702_12fb,
    0x0000_0000_a7e4_251b,
    0x0000_0009_ec76_2cc0,
    0x0000_000d_8a3a_1f48,
    0x0000_000d_b680_f346,
    0x0000_0004_a1e9_3a9d,
    0x0000_0006_38dd_c04f,
    0x0000_0004_c2fc_c993,
    0x0000_0000_1ef2_8c95,
    0x0000_000b_f0d9_792d,
    0x0000_0006_d275_57c3,
    0x0000_0006_23f9_77f4,
    0x0000_0003_5b43_be57,
    0x0000_000b_b0c4_28d5,
    0x0000_000a_6f01_474d,
    0x0000_0005_a70c_9749,
    0x0000_0002_0dda_bc3b,
    0x0000_0002_eabd_78cf,
    0x0000_0009_0aa1_8f88,
    0x0000_000a_9ea8_9350,
    0x0000_0003_cdb3_9b22,
    0x0000_0008_39a0_8f34,
    0x0000_0001_69bb_814e,
    0x0000_0001_a575_ab08,
    0x0000_000a_04d3_d5a2,
    0x0000_000b_f790_2f2b,
    0x0000_0000_95a5_e65c,
    0x0000_0009_2e8f_ce94,
    0x0000_0006_7ef4_8d12,
    0x0000_0006_400d_bcac,
    0x0000_000b_12d8_fb9f,
    0x0000_0000_347f_45d3,
    0x0000_000b_3582_6f56,
    0x0000_000c_546a_c6e4,
    0x0000_0008_1cc3_5b66,
    0x0000_0004_1d14_bd57,
    0x0000_0000_c052_b168,
    0x0000_0007_d6ce_5018,
    0x0000_000a_b4ed_5ede,
    0x0000_0005_af81_7119,
    0x0000_000d_1454_b182,
    0x0000_0002_badb_090b,
    0x0000_0000_3fcb_4c0c,
    0x0000_0002_f1c2_8fd8,
    0x0000_0009_3608_c6f7,
    0x0000_0004_c93b_a2b5,
    0x0000_0000_7d95_0a5d,
    0x0000_000e_54b3_d3fc,
    0x0000_0001_5560_cf9d,
    0x0000_0001_89e4_958a,
    0x0000_0006_2140_e9d2,
    0x0000_0007_23bc_1cdb,
    0x0000_0002_063f_26fa,
    0x0000_000f_a08a_b19f,
    0x0000_0007_9556_41db,
    0x0000_0006_46b0_1daa,
    0x0000_0007_1cd4_27cc,
    0x0000_0000_9a42_f7d4,
    0x0000_0007_17ed_c643,
    0x0000_0001_5eb9_4367,
    0x0000_0008_392e_6bb2,
    0x0000_0008_3240_8542,
    0x0000_0002_b9b8_74be,
    0x0000_000b_21f4_730d,
    0x0000_000b_5d8f_24c9,
    0x0000_0007_dbaf_6931,
    0x0000_0001_b4e3_3629,
    0x0000_0001_3452_e710,
    0x0000_000e_974a_f612,
    0x0000_0001_df61_d29a,
    0x0000_0009_9f25_32ad,
    0x0000_000e_50ec_71b4,
    0x0000_0005_df0a_36e8,
    0x0000_0004_934e_4cea,
    0x0000_000e_34a0_b4bd,
    0x0000_000b_7b26_b588,
    0x0000_0000_f255_118d,
    0x0000_000d_0c8f_a31e,
    0x0000_0000_6a50_c94f,
    0x0000_000f_28aa_9f06,
    0x0000_0001_31d1_94d8,
    0x0000_0006_22e3_da79,
    0x0000_000a_c747_8303,
    0x0000_000c_8f25_21d7,
    0x0000_0006_c9c8_81f5,
    0x0000_0004_9e38_b60a,
    0x0000_0005_13d8_df65,
    0x0000_000d_7c2b_0785,
    0x0000_0009_f6f9_d75a,
    0x0000_0009_f696_6020,
    0x0000_0001_e1a5_4e33,
    0x0000_000c_04d6_3419,
    0x0000_0009_46e0_4cd7,
    0x0000_0001_bdac_5902,
    0x0000_0005_6469_b830,
    0x0000_000f_fad5_9569,
    0x0000_0008_6970_e7d8,
    0x0000_0008_a4b4_1e12,
    0x0000_000a_d468_8e3b,
    0x0000_0008_5f8f_5df4,
    0x0000_000d_833a_0893,
    0x0000_0002_a36f_dd7c,
    0x0000_000d_6a85_7cf2,
    0x0000_0008_829b_c35c,
    0x0000_0005_e50d_79bc,
    0x0000_000f_bb80_35e4,
    0x0000_000c_1a95_bebf,
    0x0000_0000_36b0_baf8,
    0x0000_000e_0da9_64ea,
    0x0000_000b_6483_689b,
    0x0000_0007_c8e2_f4c1,
    0x0000_0005_b856_a23b,
    0x0000_0002_fc18_3995,
    0x0000_000e_914b_6d70,
    0x0000_000b_3104_1969,
    0x0000_0001_bb47_8493,
    0x0000_0000_63e2_b456,
    0x0000_000f_2a08_2b9c,
    0x0000_0008_e5e6_46ea,
    0x0000_0000_8172_f8f6,
    0x0000_0000_dacd_923e,
    0x0000_000e_5dcf_0e2e,
    0x0000_000b_f944_6bae,
    0x0000_0004_822d_50d1,
    0x0000_0002_6e71_0bf5,
    0x0000_000b_90ba_2a24,
    0x0000_000f_3b25_aa73,
    0x0000_0008_09ad_589b,
    0x0000_0009_4cc1_e254,
    0x0000_0005_334a_3adb,
    0x0000_0005_9288_6b2f,
    0x0000_000b_f647_04aa,
    0x0000_0005_66db_f24c,
    0x0000_0007_2203_e692,
    0x0000_0006_4e61_e809,
    0x0000_000d_7259_aad6,
    0x0000_0007_b924_aedc,
    0x0000_0002_df21_84e8,
    0x0000_0003_53d1_eca7,
    0x0000_000f_ce30_d7ce,
    0x0000_000f_7b0f_436e,
    0x0000_0005_7e8d_8f68,
    0x0000_0008_c79e_60db,
    0x0000_0009_c836_2b2b,
    0x0000_0006_3a58_04f2,
    0x0000_0009_2983_53dc,
    0x0000_0006_f98a_71c8,
    0x0000_000a_5731_f693,
    0x0000_0002_1ca5_c870,
    0x0000_0001_c210_7fd3,
    0x0000_0006_181f_6c39,
    0x0000_0001_9e57_4304,
    0x0000_0003_2993_7606,
    0x0000_0000_43d5_c70d,
    0x0000_0009_b18f_f162,
    0x0000_0008_e2cc_febf,
    0x0000_0007_2b7b_9b54,
    0x0000_0009_b71f_4f3c,
    0x0000_0009_35d7_393e,
    0x0000_0006_5938_881a,
    0x0000_0006_a5bd_6f2d,
    0x0000_000a_1978_3306,
    0x0000_000e_6472_f4d7,
    0x0000_0008_1163_df5a,
    0x0000_000a_838e_1cbd,
    0x0000_0009_8274_8477,
    0x0000_0000_50c5_4feb,
    0x0000_0000_d82f_bb58,
    0x0000_0002_c4c7_2799,
    0x0000_0009_7d25_9ad6,
    0x0000_0002_2d9a_43ed,
    0x0000_000f_db16_2a9f,
    0x0000_0000_cb4a_727d,
    0x0000_0004_fae2_e371,
    0x0000_0005_35b5_be8b,
    0x0000_0004_8795_908a,
    0x0000_000c_e7c1_8962,
    0x0000_0004_ea15_4d80,
    0x0000_0005_0c06_4889,
    0x0000_0008_d97f_c75d,
    0x0000_000c_8bd9_ec61,
    0x0000_0008_3ee8_e8bb,
    0x0000_000c_8431_419a,
    0x0000_0001_aa78_079d,
    0x0000_0008_111a_a4a5,
    0x0000_000d_fa3a_69fe,
    0x0000_0005_1630_d83f,
    0x0000_0002_d930_fb3f,
    0x0000_0002_1331_16e5,
    0x0000_000a_e539_5522,
    0x0000_000b_c07a_4e8a,
    0x0000_0005_7bf0_8ba0,
    0x0000_0006_cb18_036a,
    0x0000_000f_0e2e_4b75,
    0x0000_0003_eb69_2b6f,
    0x0000_000d_8178_a3fa,
    0x0000_0002_38cc_e6a6,
    0x0000_000e_97d5_cdd7,
    0x0000_000f_e10d_8d5e,
    0x0000_000b_3958_4a1d,
    0x0000_000c_a035_36fd,
    0x0000_000a_a61f_3998,
    0x0000_0007_2ff2_3ec2,
    0x0000_0001_5aa7_d770,
    0x0000_0005_7a3a_1282,
    0x0000_000d_1f39_02dc,
    0x0000_0006_554c_9388,
    0x0000_000f_d012_83c7,
    0x0000_000e_8baa_42c5,
    0x0000_0007_2cee_6adf,
    0x0000_000f_6614_b3fa,
    0x0000_0009_5c37_78a2,
    0x0000_0007_da4c_ea7a,
    0x0000_000d_18a5_912c,
    0x0000_000d_1164_26e5,
    0x0000_0002_7c17_bc1c,
    0x0000_000b_95b5_3bc1,
    0x0000_000c_8f93_7a05,
    0x0000_000e_d220_c9bd,
    0x0000_0000_c97d_72ab,
    0x0000_0008_fb12_17ae,
    0x0000_0002_5ca8_a5a1,
    0x0000_000b_261b_871b,
    0x0000_0001_bef0_a056,
    0x0000_0008_06a5_1179,
    0x0000_000e_ed24_9145,
    0x0000_0003_f82a_eceb,
    0x0000_000c_c56e_9acf,
    0x0000_0002_e78d_01eb,
    0x0000_0001_02ce_e17f,
    0x0000_0003_7caa_d3d5,
    0x0000_0001_6ac5_b1ee,
    0x0000_0002_af16_4ece,
    0x0000_000d_4cd8_1dc9,
    0x0000_0001_2263_a7e7,
    0x0000_0005_7ac7_d117,
    0x0000_0009_391d_9740,
    0x0000_0007_aeda_a77f,
    0x0000_0009_675a_3c72,
    0x0000_0002_77f2_5191,
    0x0000_000e_bb6e_64b9,
    0x0000_0007_ad3e_f747,
    0x0000_0001_2759_b181,
    0x0000_0009_4825_7d4d,
    0x0000_000b_63a8_50f6,
    0x0000_0003_a52a_8f75,
    0x0000_0004_a019_532c,
    0x0000_000a_021a_7529,
    0x0000_000c_c661_876d,
    0x0000_0004_085a_fd05,
    0x0000_000e_7048_e089,
    0x0000_0003_f979_cdc6,
    0x0000_000d_9da9_071b,
    0x0000_000e_d2fc_5b68,
    0x0000_0007_9d64_c3a1,
    0x0000_000f_d44e_2361,
    0x0000_0008_eea4_6a74,
    0x0000_0004_2233_b9c2,
    0x0000_000a_e4d1_765d,
    0x0000_0007_303a_094c,
    0x0000_0002_d703_3abe,
    0x0000_0003_dcc2_b0b4,
    0x0000_0000_f096_7d09,
    0x0000_0000_6f0c_d7de,
    0x0000_0000_9807_aca0,
    0x0000_0003_a295_cad3,
    0x0000_0002_b106_b202,
    0x0000_0003_f38a_828e,
    0x0000_0007_8af4_6596,
    0x0000_000b_da2d_c713,
    0x0000_0009_a8c8_c9d9,
    0x0000_0006_a0f2_ddce,
    0x0000_000a_76af_6fe2,
    0x0000_0000_86f6_6fa4,
    0x0000_000d_52d6_3f8d,
    0x0000_0008_9f7a_6e73,
    0x0000_000c_c6b2_3362,
    0x0000_000b_4ebf_3c39,
    0x0000_0005_64f3_00fa,
    0x0000_000e_8de3_a706,
    0x0000_0007_9a03_3b61,
    0x0000_0007_65e1_60c5,
    0x0000_000a_266a_4f85,
    0x0000_000a_68c3_8c24,
    0x0000_000d_ca07_11fb,
    0x0000_0008_5fba_85ba,
    0x0000_0003_7a20_7b46,
    0x0000_0001_58fc_c4d0,
    0x0000_0000_569d_79b3,
    0x0000_0007_b1a2_5555,
    0x0000_000a_8ae2_2468,
    0x0000_0007_c592_bdfd,
    0x0000_0000_c59a_5f66,
    0x0000_000b_1115_daa3,
    0x0000_000f_17c8_7177,
    0x0000_0006_769d_766b,
    0x0000_0002_b637_356d,
    0x0000_0001_3d86_85ac,
    0x0000_000f_24cb_6ec0,
    0x0000_0000_bd0b_56d1,
    0x0000_0004_2ff0_e26d,
    0x0000_000b_4160_9267,
    0x0000_0009_6f95_18af,
    0x0000_000c_56f9_6636,
    0x0000_0004_a8e1_0349,
    0x0000_0008_6351_2171,
    0x0000_000e_a455_d86c,
    0x0000_000b_d0e2_5279,
    0x0000_000e_65e3_f761,
    0x0000_0003_6c84_a922,
    0x0000_0008_5fd1_b38f,
    0x0000_0006_57c9_1539,
    0x0000_0001_5033_fe04,
    0x0000_0000_9051_c921,
    0x0000_000a_b27d_80d8,
    0x0000_000f_92f7_d0a1,
    0x0000_0008_eb6b_b737,
    0x0000_0001_0b5b_0f63,
    0x0000_0006_c9c7_ad63,
    0x0000_000f_66fe_70ae,
    0x0000_000c_a579_bd92,
    0x0000_0009_5619_8e4d,
    0x0000_0002_9e44_05e5,
    0x0000_000e_44eb_885c,
    0x0000_0004_1612_456c,
    0x0000_000e_a45e_0abf,
    0x0000_000d_3265_29bd,
    0x0000_0007_b2c3_3cef,
    0x0000_0008_0bc9_b558,
    0x0000_0007_169b_9740,
    0x0000_000c_37f9_9209,
    0x0000_0003_1ff6_dab9,
    0x0000_000c_7951_90ed,
    0x0000_000a_7636_e95f,
    0x0000_0009_df07_5841,
    0x0000_0005_5a08_3932,
    0x0000_000a_7cbd_f630,
    0x0000_0004_09ea_4ef0,
    0x0000_0009_2a19_91b6,
    0x0000_0004_b078_dee9,
    0x0000_000a_e18c_e9e4,
    0x0000_0005_a6e1_ef35,
    0x0000_0001_a403_bd59,
    0x0000_0003_1ea7_0a83,
    0x0000_0002_bc3c_4f3a,
    0x0000_0005_c921_b3cb,
    0x0000_0000_42da_05c5,
    0x0000_0001_f667_d16b,
    0x0000_0004_16a3_68cf,
    0x0000_000f_bc0a_7a3b,
    0x0000_0009_419f_0c7c,
    0x0000_0008_1be2_fa03,
    0x0000_0003_4e2c_172f,
    0x0000_0002_8648_d8ae,
    0x0000_000c_7acb_b885,
    0x0000_0004_5f31_eb6a,
    0x0000_000d_1cfc_0a7b,
    0x0000_0004_2c4d_260d,
    0x0000_000c_f658_4097,
    0x0000_0009_4b13_2b14,
    0x0000_0003_c5c5_df75,
    0x0000_0008_ae59_6fef,
    0x0000_000a_ea80_54eb,
    0x0000_0000_ae9c_c573,
    0x0000_0004_96fb_731b,
    0x0000_000e_bf10_5662,
    0x0000_000a_f9c8_3a37,
    0x0000_000c_0d64_cd6b,
    0x0000_0007_b608_159a,
    0x0000_000e_7443_1642,
    0x0000_000d_6fb9_d900,
    0x0000_0002_91e9_9de0,
    0x0000_0001_0500_ba9a,
    0x0000_0005_cd05_d037,
    0x0000_000a_8725_4fb2,
    0x0000_0009_d782_4a37,
    0x0000_0008_b2c7_b47c,
    0x0000_0003_0c78_8145,
    0x0000_0002_f4e5_a8be,
    0x0000_000b_adb8_84da,
    0x0000_0000_26e0_d5c9,
    0x0000_0006_fdba_a32e,
    0x0000_0003_4758_eb31,
    0x0000_0005_65cd_1b4f,
    0x0000_0002_bfd9_0fb0,
    0x0000_0000_9305_2a6b,
    0x0000_000d_3c13_c4b9,
    0x0000_0002_daea_43bf,
    0x0000_000a_2797_62bc,
    0x0000_000f_1bd9_f22c,
    0x0000_0004_b7fe_c94f,
    0x0000_0005_4576_1d5a,
    0x0000_0007_327d_f411,
    0x0000_0001_b52a_442e,
    0x0000_0004_9b0c_e108,
    0x0000_0002_4c76_4bc8,
    0x0000_0003_7456_3045,
    0x0000_000a_3e8f_91c6,
    0x0000_0000_e6bd_2241,
    0x0000_000e_0e52_ee3c,
    0x0000_0000_7e8e_3caa,
    0x0000_0009_6c2b_7372,
    0x0000_0003_3acb_dfda,
    0x0000_000b_15d9_1e54,
    0x0000_0004_6475_9ac1,
    0x0000_0006_886a_1998,
    0x0000_0005_7f5d_3958,
    0x0000_0005_a1f5_c1f5,
    0x0000_0000_b581_58ad,
    0x0000_000e_7120_53fb,
    0x0000_0005_352d_db25,
    0x0000_0004_14b9_8ea0,
    0x0000_0007_4f89_f546,
    0x0000_0003_8a56_b3c3,
    0x0000_0003_8db0_dc17,
    0x0000_000a_a016_a755,
    0x0000_000d_c723_66f5,
    0x0000_0000_cee9_3d75,
    0x0000_000b_2fe7_a56b,
    0x0000_000a_847e_d390,
    0x0000_0008_713e_f88c,
    0x0000_000a_217c_c861,
    0x0000_0008_bca2_5d7b,
    0x0000_0004_5552_6818,
    0x0000_000e_a3a7_a180,
    0x0000_000a_9536_e5e0,
    0x0000_0009_b64a_1975,
    0x0000_0005_bfc7_56bc,
    0x0000_0000_46aa_169b,
    0x0000_0005_3a17_f76f,
    0x0000_0004_d681_5274,
    0x0000_000c_ca9c_f3f6,
    0x0000_0004_013f_cb8b,
    0x0000_0003_d26c_dfa5,
    0x0000_0005_7862_31f7,
    0x0000_0007_d4ab_09ab,
    0x0000_0009_60b5_ffbc,
    0x0000_0008_914d_f0d4,
    0x0000_0002_fc6f_2213,
    0x0000_000a_c235_637e,
    0x0000_0001_51b2_8ed3,
    0x0000_0004_6f79_b6db,
    0x0000_0001_382e_0c9f,
    0x0000_0005_3abf_983a,
    0x0000_0003_83c4_7ade,
    0x0000_0003_fcf8_8978,
    0x0000_000e_b907_9df7,
    0x0000_0000_9af0_714d,
    0x0000_000d_a19d_1bb7,
    0x0000_0009_a027_49f8,
    0x0000_0001_c62d_ab9b,
    0x0000_0001_a137_e44b,
    0x0000_0002_8677_18c7,
    0x0000_0003_5815_525b,
    0x0000_0007_cd35_c550,
    0x0000_0002_164f_73a0,
    0x0000_000e_8b77_2fe0,
];

/// A lookup table for O(1) candidate generation based on the pigeonhole principle.
pub struct QuickDecode {
    chunk_offsets: [[u16; 513]; 4],
    chunk_ids: [[u16; 587]; 4],
}

impl QuickDecode {
    /// Builds the LUT. This should be called exactly ONCE at application startup.
    pub fn new() -> Self {
        let mut offsets = [[0u16; 513]; 4];
        let mut ids = [[0u16; 587]; 4];

        for &code in &TAG36H11_CODES {
            for (i, row) in offsets.iter_mut().enumerate().take(4) {
                let val = ((code >> (i * 9)) & 0x1FF) as usize;
                row[val + 1] += 1;
            }
        }

        for row in offsets.iter_mut().take(4) {
            for j in 0..512 {
                row[j + 1] += row[j];
            }
        }

        let mut cursors = offsets;
        for (id, &code) in TAG36H11_CODES.iter().enumerate() {
            for i in 0..4 {
                let val = ((code >> (i * 9)) & 0x1FF) as usize;
                let write_pos = cursors[i][val] as usize;
                ids[i][write_pos] = id as u16;
                cursors[i][val] += 1;
            }
        }

        Self {
            chunk_offsets: offsets,
            chunk_ids: ids,
        }
    }

    /// Decodes a 36-bit payload, returning (`TagID`, `HammingDistance`) if found.
    pub fn decode(&self, observed_code: u64) -> Option<(u16, u8)> {
        for i in 0..4 {
            let val = ((observed_code >> (i * 9)) & 0x1FF) as usize;

            let start = self.chunk_offsets[i][val] as usize;
            let end = self.chunk_offsets[i][val + 1] as usize;

            for &id in &self.chunk_ids[i][start..end] {
                let perfect_code = TAG36H11_CODES[id as usize];
                let dist = (observed_code ^ perfect_code).count_ones() as u8;

                if dist <= TAG36H11_MAX_HAMMING as u8 {
                    return Some((id, dist));
                }
            }
        }
        None
    }
}

/// The final output of the `AprilTag` pipeline.
/// `repr(C, packed)` ensures it can be safely passed over FFI or DMA to other system components.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Serialize)]
pub struct AprilTagDetection {
    pub id: u16,         // The decoded tag ID (0-586 for 36h11)
    pub hamming: u8,     // Number of bit errors corrected (0, 1, or 2)
    pub rotation: u8,    // Physical rotation in 90-deg increments (0-3)
    pub confidence: f32, // The decision margin from the GrayModel
    pub center_x: f32,   // Center pixel X
    pub center_y: f32,   // Center pixel Y

    // 3D Translation Vector (tvec) from PnP Solver
    pub tx: f32, // Translation X in camera frame (mm, lateral)
    pub ty: f32, // Translation Y in camera frame (mm, vertical)
    pub tz: f32, // Translation Z in camera frame (mm, depth/forward)

    // 3D Rotation (Euler angles derived from rvec)
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,

    pub distance_mm: f32, // Euclidean distance: sqrt(tx^2 + ty^2 + tz^2)
}

pub struct Homography {
    pub h: SMatrix<f32, 3, 3>,
}

impl Homography {
    pub fn compute(corners: &[[f32; 2]; 4]) -> Option<Self> {
        let ideal = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let mut a = SMatrix::<f32, 8, 8>::zeros();
        let mut b = SVector::<f32, 8>::zeros();

        for i in 0..4 {
            let (ix, iy) = ideal[i];
            let px = corners[i][0];
            let py = corners[i][1];

            a[(i * 2, 0)] = ix;
            a[(i * 2, 1)] = iy;
            a[(i * 2, 2)] = 1.0;
            a[(i * 2, 6)] = -ix * px;
            a[(i * 2, 7)] = -iy * px;
            b[i * 2] = px;

            a[(i * 2 + 1, 3)] = ix;
            a[(i * 2 + 1, 4)] = iy;
            a[(i * 2 + 1, 5)] = 1.0;
            a[(i * 2 + 1, 6)] = -ix * py;
            a[(i * 2 + 1, 7)] = -iy * py;
            b[i * 2 + 1] = py;
        }

        let decomp = a.lu();
        decomp.solve(&b).map(|x| {
            let h = SMatrix::<f32, 3, 3>::new(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1.0);
            Self { h }
        })
    }

    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn project(&self, x: f32, y: f32) -> (f32, f32) {
        let h = &self.h;
        let w = h[(2, 1)].mul_add(y, h[(2, 0)] * x) + h[(2, 2)];
        let px = (h[(0, 1)].mul_add(y, h[(0, 0)] * x) + h[(0, 2)]) / w;
        let py = (h[(1, 1)].mul_add(y, h[(1, 0)] * x) + h[(1, 2)]) / w;
        (px, py)
    }
}

#[derive(Default)]
pub struct GrayModel {
    a: SMatrix<f32, 3, 3>,
    b: SVector<f32, 3>,
    c: SVector<f32, 3>,
}

impl GrayModel {
    pub fn add(&mut self, x: f32, y: f32, intensity: f32) {
        self.a[(0, 0)] = x.mul_add(x, self.a[(0, 0)]);
        self.a[(0, 1)] = x.mul_add(y, self.a[(0, 1)]);
        self.a[(0, 2)] += x;
        self.a[(1, 0)] = x.mul_add(y, self.a[(1, 0)]);
        self.a[(1, 1)] = y.mul_add(y, self.a[(1, 1)]);
        self.a[(1, 2)] += y;
        self.a[(2, 0)] += x;
        self.a[(2, 1)] += y;
        self.a[(2, 2)] += 1.0;

        self.b[0] = x.mul_add(intensity, self.b[0]);
        self.b[1] = y.mul_add(intensity, self.b[1]);
        self.b[2] += intensity;
    }

    pub fn solve(&mut self) {
        if let Some(x) = self.a.cholesky().map(|c| c.solve(&self.b)) {
            self.c = x;
        } else if let Some(x) = self.a.lu().solve(&self.b) {
            self.c = x;
        }
    }

    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn interpolate(&self, x: f32, y: f32) -> f32 {
        self.c[1].mul_add(y, self.c[0] * x) + self.c[2]
    }
}

/// Bilinear interpolation for sub-pixel accuracy
fn sample_pixel(image: &Image, px: f32, py: f32) -> Option<f32> {
    let x = px - 0.5;
    let y = py - 0.5;
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;

    if ix < 0 || ix + 1 >= image.width as i32 || iy < 0 || iy + 1 >= image.height as i32 {
        return None;
    }

    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let v00 = f32::from(image.row(iy as usize)[ix as usize]);
    let v10 = f32::from(image.row(iy as usize)[ix as usize + 1]);
    let v01 = f32::from(image.row(iy as usize + 1)[ix as usize]);
    let v11 = f32::from(image.row(iy as usize + 1)[ix as usize + 1]);

    let val = (v11 * fx).mul_add(
        fy,
        (v01 * (1.0 - fx)).mul_add(
            fy,
            (v10 * fx).mul_add(1.0 - fy, v00 * (1.0 - fx) * (1.0 - fy)),
        ),
    );

    Some(val)
}

/// Rotates a 36-bit tag payload 90 degrees clockwise.
#[inline(always)]
#[allow(clippy::inline_always)]
const fn rotate90(w: u64) -> u64 {
    ((w << 9) | (w >> 27)) & 0x0F_FFFF_FFFF
}

/// Matches the payload using the O(1) `QuickDecode` table.
/// Returns (`TagID`, `HammingDistance`, `RotationIndex`)
fn match_payload(quick_decode: &QuickDecode, mut payload: u64) -> Option<(u16, u8, u8)> {
    for rotation in 0..4 {
        if let Some((id, hamming)) = quick_decode.decode(payload) {
            return Some((id, hamming, rotation));
        }
        payload = rotate90(payload);
    }
    None
}

/// Takes a bounding quad and extracts the `AprilTag` data, returning a valid detection.
pub fn extract_detection(
    image: &Image,
    corners: &[[f32; 2]; 4],
    intrinsics: &CameraIntrinsics,
    quick_decode: &QuickDecode,
) -> Option<AprilTagDetection> {
    let mut refined_corners = *corners;
    refine_edges(image, &mut refined_corners);

    let homo = Homography::compute(&refined_corners)?;

    let (white_model, black_model) = build_and_solve_gray_models(image, &homo)?;
    let grid = sample_tag_grid(image, &homo, &white_model, &black_model)?;

    let (rcode, confidence) = extract_payload_and_confidence(&grid);
    let (id, hamming, rotation) = match_payload(quick_decode, rcode)?;

    let (corrected_homo, corrected_corners) = apply_rotation(&homo, &refined_corners, rotation);
    let (center_x, center_y) = corrected_homo.project(0.0, 0.0);

    let pose = estimate_tag_pose(&corrected_homo, &corrected_corners, intrinsics);

    Some(AprilTagDetection {
        id,
        hamming,
        rotation,
        confidence,
        center_x,
        center_y,
        tx: pose.t.x,
        ty: pose.t.y,
        tz: pose.t.z,
        yaw: pose.yaw,
        pitch: pose.pitch,
        roll: pose.roll,
        distance_mm: pose.distance_mm,
    })
}

/// Models the lighting gradient across the tag to distinguish black/white threshold.
fn build_and_solve_gray_models(image: &Image, homo: &Homography) -> Option<(GrayModel, GrayModel)> {
    let mut black_model = GrayModel::default();
    let mut white_model = GrayModel::default();

    let steps = TAG36H11_WIDTH_AT_BORDER;
    let delta = 2.0 / (steps as f32);

    for i in 0..steps {
        let t = (i as f32 + 0.5).mul_add(delta, -1.0);
        let white_offset = 1.0 / (TAG36H11_WIDTH_AT_BORDER as f32);

        let border_samples = [
            (t, -1.0 + white_offset, false),
            (t, -1.0 - white_offset, true),
            (t, 1.0 - white_offset, false),
            (t, 1.0 + white_offset, true),
            (-1.0 + white_offset, t, false),
            (-1.0 - white_offset, t, true),
            (1.0 - white_offset, t, false),
            (1.0 + white_offset, t, true),
        ];

        for &(tag_x, tag_y, is_white) in &border_samples {
            let (px, py) = homo.project(tag_x, tag_y);
            if let Some(val) = sample_pixel(image, px, py) {
                if is_white {
                    white_model.add(tag_x, tag_y, val);
                } else {
                    black_model.add(tag_x, tag_y, val);
                }
            }
        }
    }

    white_model.solve();
    black_model.solve();

    if white_model.interpolate(0.0, 0.0) < black_model.interpolate(0.0, 0.0) {
        return None;
    }

    Some((white_model, black_model))
}

/// Iterates over the 8x8 interior grid of the tag, sampling pixels against the local decision threshold.
fn sample_tag_grid(
    image: &Image,
    homo: &Homography,
    white_model: &GrayModel,
    black_model: &GrayModel,
) -> Option<[f32; 64]> {
    let mut grid = [0.0f32; 64];

    for by in 0..8 {
        for bx in 0..8 {
            let tag_x = 2.0f32.mul_add((bx as f32 + 0.5) / 8.0, -1.0);
            let tag_y = 2.0f32.mul_add((by as f32 + 0.5) / 8.0, -1.0);

            let (px, py) = homo.project(tag_x, tag_y);

            let val = sample_pixel(image, px, py)?;
            let white_thresh = white_model.interpolate(tag_x, tag_y);
            let black_thresh = black_model.interpolate(tag_x, tag_y);
            let decision_thresh = f32::midpoint(white_thresh, black_thresh);

            grid[by * 8 + bx] = val - decision_thresh;
        }
    }

    Some(grid)
}

/// Applies a Laplacian sharpening filter over the grid and extracts the 36-bit integer payload.
fn extract_payload_and_confidence(grid: &[f32; 64]) -> (u64, f32) {
    let mut rcode: u64 = 0;
    let mut white_score = 0.0;
    let mut black_score = 0.0;
    let mut white_count = 1.0;
    let mut black_count = 1.0;

    let decode_sharpening = 0.25;

    for i in 0..TAG36H11_NBITS {
        let bx = TAG36H11_BIT_X[i];
        let by = TAG36H11_BIT_Y[i];
        let idx = by * 8 + bx;

        let v = grid[idx];

        let laplacian = 4.0f32.mul_add(v, -grid[idx - 8]) // Top
            - grid[idx + 8] // Bottom
            - grid[idx - 1] // Left
            - grid[idx + 1]; // Right

        let final_val = v + decode_sharpening * laplacian;

        rcode <<= 1;
        if final_val > 0.0 {
            rcode |= 1;
            white_score += final_val;
            white_count += 1.0;
        } else {
            black_score -= final_val;
            black_count += 1.0;
        }
    }

    let confidence = (white_score / white_count).min(black_score / black_count);
    (rcode, confidence)
}

/// Returns a new homography matrix and corner array adjusted for the tag's true orientation.
fn apply_rotation(
    homo: &Homography,
    corners: &[[f32; 2]; 4],
    rotation: u8,
) -> (Homography, [[f32; 2]; 4]) {
    let theta = f32::from(rotation) * std::f32::consts::PI / 2.0;
    let c = theta.cos();
    let s = theta.sin();
    let r_mat = nalgebra::SMatrix::<f32, 3, 3>::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);

    let corrected_homo = Homography { h: homo.h * r_mat };

    let mut corrected_corners = [[0.0; 2]; 4];
    for i in 0..4 {
        corrected_corners[i] = corners[(i + rotation as usize) % 4];
    }

    (corrected_homo, corrected_corners)
}
