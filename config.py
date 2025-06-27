import torch
import albumentations as A

model = 'S2DNets'#'T1c5PDNet_clus_bias_tv'
fold = 'BrainWeb'
datasetName = 'BrainWeb'
modality = 'T2'
device = "cuda" if torch.cuda.is_available() else "cpu"
trainDir = '/home/liang/Data/BrainWeb/train/'
testDir = '/home/liang/Data/BrainWeb/train/T1_30/'
saveDir = r'/home/liang/Data/MICCAIREVIEW/%s/%s/'%(fold, model)
batchSize = 4
classNum = 4
workerNum = 2
criticNum = 5
set_height = 256
set_width = 256
learningRate = 0.001
epochs = 1500
suffix = '.png'
excel_save = False
model_save = True
LAMBDA_IDENTITY = 10
LAMBDA_GEOCONSI = 20
LAMBDA_CYCLE = 10
LAMBDA_BIAS = 1
inf = 1e-9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

checkpointBiasGenPath = r"/home/liang/Data/checkpoint/%s/%s_genBias.pth.tar"%(fold,model)
checkpointBiasCluPath = r"/home/liang/Data/checkpoint/%s/%s_cluBias.pth.tar"%(fold,model)
checkpointClusGenPath = r"/home/liang/Data/checkpoint/%s/%s_genClus.pth.tar"%(fold,model)
checkpointClusCluPath = r"/home/liang/Data/checkpoint/%s/%s_cluClus.pth.tar"%(fold,model)
checkpointGen = r"/home/liang/Data/checkpoint/%s/%s_gen.pth.tar"%(fold,model)
checkpointClu = r"/home/liang/Data/checkpoint/%s/%s_clu.pth.tar"%(fold,model)
dir_excel = r"/home/liang/Data/%s/ModelsPerformance/Excel/%s_%s.xlsx"%(fold,fold,model)

testcheckpointGen = checkpointBiasGenPath

transforms = A.Compose([A.HorizontalFlip(p=0.5)], additional_targets={"clearImg":"image"},)

