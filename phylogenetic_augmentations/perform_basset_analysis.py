# ####################################################################################################################
# perform_basset_analysis.py
#
# Train model using the Basset data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import numpy as np
import utils
import models_basset as models

# ====================================================================================================================
# Arguments
# ====================================================================================================================
model_type = sys.argv[1]
replicate = sys.argv[2]
use_homologs = bool(int(sys.argv[3]))
sample_fraction = float(sys.argv[4])
gpu_id = sys.argv[5]

file_folder = "../process_data/basset/output/"
homolog_folder = "../process_data/basset/output/orthologs/per_species_fa/"
output_folder = "./output_basset/"
sequence_size = 600
tasks = ["8988T", "AoSMC", "Chorion", "CLL", "Fibrobl", "FibroP", "Gliobla", "GM12891", "GM12892", "GM18507", "GM19238", "GM19239", "GM19240", "H9ES",
         "HeLa-S3_IFNa4h", "Hepatocytes", "HPDE6-E6E7", "HSMM_emb", "HTR8svn", "Huh-7.5", "Huh-7", "iPS", "Ishikawa_Estradiol", "Ishikawa_4OHTAM",
         "LNCaP_androgen", "MCF-7_Hypoxia", "Medullo", "Melano", "Myometr", "Osteobl", "PanIsletD", "PanIslets", "pHTE", "ProgFib", "RWPE1", "Stellate",
         "T-47D", "CD4_Th0", "Urothelia", "Urothelia_UT189", "AG04449", "AG04450", "AG09309", "AG09319", "AG10803", "AoAF", "BE2_C", "BJ", "Caco-2", "CD20+",
         "CD34+", "CMK", "GM06990", "GM12864", "GM12865", "H7-hESC", "HAc", "HAEpiC", "HA-h", "HA-sp", "HBMEC", "HCF", "HCFaa", "HCM", "HConF", "HCPEpiC", "HCT-116",
         "HEEpiC", "HFF", "HFF-Myc", "HGF", "HIPEpiC", "HL-60", "HMF", "HMVEC-dAd", "HMVEC-dBl-Ad", "HMVEC-dBl-Neo", "HMVEC-dLy-Ad", "HMVEC-dLy-Neo", "HMVEC-dNeo",
         "HMVEC-LBl", "HMVEC-LLy", "HNPCEpiC", "HPAEC", "HPAF", "HPdLF", "HPF", "HRCEpiC", "HRE", "HRGEC", "HRPEpiC", "HVMF", "Jurkat", "Monocytes-CD14+", "NB4", "NH-A",
         "NHDF-Ad", "NHDF-neo", "NHLF", "NT2-D1", "PANC-1", "PrEC", "RPTEC", "SAEC", "SKMC", "SK-N-MC", "SK-N-SH_RA", "Th2", "WERI-Rb-1", "WI-38", "WI-38_4OHTAM", "A549",
         "GM12878", "H1-hESC", "HeLa-S3", "HepG2", "HMEC", "HSMM", "HSMMtube", "HUVEC", "K562", "LNCaP", "MCF-7", "NHEK", "Th1", "LNG.IMR90", "ESC.H9", "ESC.H1",
         "IPSC.DF.6.9", "IPSC.DF.19.11", "ESDR.H1.NEUR.PROG", "ESDR.H1.BMP4.MESO", "ESDR.H1.BMP4.TROP", "ESDR.H1.MSC", "BLD.CD3.PPC", "BLD.CD3.CPC", "BLD.CD14.PC",
         "BLD.MOB.CD34.PC.M", "BLD.MOB.CD34.PC.F", "BLD.CD19.PPC", "BLD.CD56.PC", "SKIN.PEN.FRSK.FIB.01", "SKIN.PEN.FRSK.FIB.02", "SKIN.PEN.FRSK.MEL.01",
         "SKIN.PEN.FRSK.KER.02", "BRST.HMEC.35", "THYM.FET", "BRN.FET.F", "BRN.FET.M", "MUS.PSOAS", "MUS.TRNK.FET", "MUS.LEG.FET", "HRT.FET", "GI.STMC.FET",
         "GI.S.INT.FET", "GI.L.INT.FET", "GI.S.INT", "GI.STMC.GAST", "KID.FET", "LNG.FET", "OVRY", "ADRL.GLND.FET", "PLCNT.FET", "PANC"]

# ====================================================================================================================
# Main code
# ====================================================================================================================

num_samples_train = utils.count_lines_in_file(
    file_folder + "Sequences_activity_Train.txt") - 1

filtered_indices = None
if int(sample_fraction) < 1:
    reduced_num_samples_train = int(num_samples_train * sample_fraction)
    filtered_indices = np.random.choice(
        list(range(num_samples_train)), reduced_num_samples_train, replace=False)

models.train_basset(use_homologs, sample_fraction, replicate, file_folder,
                    homolog_folder, output_folder, tasks, sequence_size, filtered_indices, model_type, gpu_id)

models.fine_tune_basset(use_homologs, sample_fraction, replicate, file_folder,
                        homolog_folder, output_folder, tasks, sequence_size, filtered_indices, model_type, gpu_id)
