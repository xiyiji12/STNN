import scipy.io as sio
from collections import defaultdict
import scipy.signal
from model import *


# 陷波滤波器：用于去除特定频率的噪声，例如工频干扰
def notch_filter(signal, f_R, fs):
    B, A = scipy.signal.iirnotch(f_R, int(f_R / 10), fs)
    return scipy.signal.lfilter(B, A, signal, axis=0)


######################################################################
# 设计一个五阶 Butterworth 带通滤波器。
def bandpass(signal, band, fs):
    B, A = scipy.signal.butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return scipy.signal.lfilter(B, A, signal, axis=0)


######################################################################
# 对数据和其对应的标签进行随机打乱
def shuffled_data(data, label):
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation, :, :]
    shuffled_label = label[permutation, :]
    return shuffled_data, shuffled_label


##################################################################################
def train_data_extraction_from_mat(Subject):  # 加载受试者的数据文件。
    # 加载为 Python 字典 path_mat。字典中的每个键对应 .mat 文件中的一个变量名。
    path_mat = sio.loadmat('data/' + 'Subject_' + Subject + '_' + 'Train' + '.mat')  # 读取受试者训练数据集
    Flashings = path_mat['Flashing'];  # 标识显示的网格行或列是否正在闪烁。通常是一个二维数组，行数对应于时间步（即不同的时刻），列数则对应不同的刺激目标（如行或列）。
    # （85，7794），85次实验（字符），每次实验重复15次,每次闪12次，闪烁100ms+间隔75ms，240hz采样率：代表是否闪烁：0/1
    print('Flashings', Flashings.shape)
    # （85，7794，64）64个脑电通道数
    Signals = path_mat['Signal'];  # 原始 EEG 数据，通常是形状为（通道数 × 时间步数）的矩阵
    print('Signals', Signals.shape)
    # （85，7794）具体闪烁的行或列1~12
    StimulusCodes = path_mat['StimulusCode'];  # 标识在每个时间点屏幕上哪个行或列正在闪烁。 1 到 6 代表行闪烁，7 到 12 代表列闪烁
    print('StimulusCodes', StimulusCodes.shape)
    # （85，7794）闪烁的字符是不是目标的，0/1标识
    StimulusTypes = path_mat['StimulusType'];  # 标识当前时间步的刺激是否是目标（用户注意的）刺激
    print('StimulusTypes', StimulusTypes)
    # 受试者的目标字符 VGREAAH8TVRHBYN_UGCOLO4EUERDOOHCIFOMDNU6LQCPKEIREKOYRQIDJXPBKOJDWZEUEWWFOEBHXTQTTZUMO
    TargetChars = path_mat['TargetChar'][0]  # 实验中受试者的目标字符
    print('TargetChars', TargetChars)

    return Flashings, Signals, StimulusCodes, StimulusTypes, TargetChars


######################################################################
# 找出在 Flashing 中一次闪烁事件的连续的 24 个样本点的最后一个位置索引。
def ERP_location(Flashing):
    loactions = [];
    Counts = 0
    for loaction, trigger in enumerate(Flashing):
        if trigger == int(1):
            Counts += 1
            if Counts == 24:  # 一行/一列闪100ms，采样率240hz，一次闪烁对应的样本点为0.1s x 240hz=24个
                loactions.append(loaction);
                Counts = 0
    return np.array(loactions)


######################################################################
# 在 P300 拼写器的字母矩阵中找到目标字符的行列位置，并返回标识它们的编码值
def Single_trial_target(TargetChar):
    sgl_Target = []
    screen = [['A', 'B', 'C', 'D', 'E', 'F'],
              ['G', 'H', 'I', 'J', 'K', 'L'],
              ['M', 'N', 'O', 'P', 'Q', 'R'],
              ['S', 'T', 'U', 'V', 'W', 'X'],
              ['Y', 'Z', '1', '2', '3', '4'],
              ['5', '6', '7', '8', '9', '_']]
    for i in range(0, 6):
        for j in range(0, 6):
            if TargetChar == screen[i][j]:
                sgl_Target += [i + 7, j + 1]
    return sgl_Target


######################################################################
# 根据刺激编码，将指定位置的刺激事件分组并存储在一个字典中，loactions 中的每个整数通常对应一个位置或时间点。
def Single_local_information(loactions, StimulusCode):
    loca_dicts = defaultdict(list)  # 初始化一个字典
    for i, code in enumerate(StimulusCode[loactions]):  # 遍历 loactions 中的每个位置和对应的刺激编码，迭代每个位置的编码，并返回位置索引 i 和对应的编码 code。
        if int(code) == 1: loca_dicts[1].append(
            loactions[i])  # 比如12: [11] 表示在 StimulusCode 中编码为 12 的位置是在 loactions 中索引为 11 的位置。
        if int(code) == 2: loca_dicts[2].append(loactions[i])
        if int(code) == 3: loca_dicts[3].append(loactions[i])
        if int(code) == 4: loca_dicts[4].append(loactions[i])
        if int(code) == 5: loca_dicts[5].append(loactions[i])
        if int(code) == 6: loca_dicts[6].append(loactions[i])
        if int(code) == 7: loca_dicts[7].append(loactions[i])
        if int(code) == 8: loca_dicts[8].append(loactions[i])
        if int(code) == 9: loca_dicts[9].append(loactions[i])
        if int(code) == 10: loca_dicts[10].append(loactions[i])
        if int(code) == 11: loca_dicts[11].append(loactions[i])
        if int(code) == 12: loca_dicts[12].append(loactions[i])

    return loca_dicts


######################################################################
# 从 EEG 信号中提取事件相关电位（ERP）数据，并按给定的采样数进行重采样。
def ERP_extraction(Signal, loca_dict, samps):
    seq = 5  # 用于平均的 ERP 片段数量。
    ERPs = []
    for response in loca_dict:  # 遍历 loca_dict 中的每个 response，即事件发生的起始时间点。
        res_start = response + 1;
        res_end = response + 1 + 120  # 定义每个响应的起止点，120 是一个窗口大小，常用于 ERP 分析（例如 1 秒的时间窗在 120 Hz 的采样率）。
        ERP = scipy.signal.resample(Signal[res_start:res_end, :],
                                    samps)  # 重采样信号片段，将其集中在 res_start 和 res_end 的窗内，并将其长度调整为 samps。
        ERPs += [ERP]
    ERPs = np.array(ERPs)  # 将片段列表转换为 NumPy 数组。
    ###
    seqs = np.array(range(int(seq)))  # 生成一组索引
    ERPs = ERPs[seqs, :, :]  # 选择用于计算平均的 ERP 片段。
    ###
    ERP = np.mean(ERPs, axis=0)  # 在所选择的片段上进行平均，得到单一的 ERP 数据。
    return ERP  # 单个 ERP 均值矩阵，其形式为（通道数 × 重采样后的样本数），代表所有相应 ERP 数据的平均。


######################################################################
# 从 EEG 信号中提取事件相关电位（ERP），并将它们划分为目标事件相关电位（Target ERPs）和非目标事件相关电位（Non-Target ERPs）
def Single_trial_ERP(Signal, sgl_Target, loca_dicts, samps):  # 完整的EEG信号数据，一个包含目标刺激的行和列编码的列表例如 [9, 4]，事件位置字典，
    Target_ERPs = [];
    NonTarget_ERPs = []
    Tar_0 = sgl_Target[0];  # 获取目标行的编码。
    Tar_1 = sgl_Target[1]  # 获取目标列的编码。
    for i in range(1, 13):  # ：遍历 1 到 12 的编码
        X_ERP = ERP_extraction(Signal, loca_dicts[i], samps)  # 提取ERP
        # 如果当前编码 i 属于目标行或列编码，把相关的 X_ERP 添加到 Target_ERPs。
        if i == Tar_0:
            Target_ERPs.append(X_ERP)
        elif i == Tar_1:
            Target_ERPs.append(X_ERP)
        else:
            NonTarget_ERPs.append(X_ERP)
    Target_ERPs = np.array(Target_ERPs);
    NonTarget_ERPs = np.array(NonTarget_ERPs)
    return Target_ERPs, NonTarget_ERPs


######################################################################
def train_data_and_label(Subject, samps):
    Flashings, Signals, StimulusCodes, StimulusTypes, TargetChars = train_data_extraction_from_mat(Subject)
    Target = [];
    NonTarget = []
    for i in range(len(Flashings)):  # 遍历每个闪烁事件，提取和处理数据
        ##Flashings 85*7794  Flashing 7794；Signals 85*7794*64  Signal 7794*64；
        Flashing = Flashings[i];
        Signal = Signals[i];
        StimulusCode = StimulusCodes[i];
        TargetChar = TargetChars[i]
        # 对信号进行带通滤波，仅保留 0.1 Hz 至 20 Hz 的频率，采样率为 240 Hz。此步骤用于去除噪声并保留 P300 信号频段。
        Signal = bandpass(Signal, [0.1, 20.0], 240)
        loactions = ERP_location(Flashing)  # 获取ERP的位置
        sgl_Target = Single_trial_target(TargetChar)  # 获取当前目标字符的单次试验信息返回他的行和列编码，比如p返回[9, 4]，2维度
        loca_dicts = Single_local_information(loactions, StimulusCode)  # 返回字典，键为刺激编码（1到12），值为一次闪烁的样本点的最后一个索引。
        Target_ERPs, NonTarget_ERPs = Single_trial_ERP(Signal, sgl_Target, loca_dicts,
                                                       samps)  # 从 EEG 信号中提取事件相关电位（ERP）并分类
        Target += [Target_ERPs];
        NonTarget += [NonTarget_ERPs]
    Target = np.array(Target);
    NonTarget = np.array(NonTarget)
    Target = Target.reshape(-1, samps, 64)  # 画图
    NonTarget = NonTarget.reshape(-1, samps, 64)
    Target_label = np.ones((Target.shape[0], 1), dtype=np.int)
    NonTarget_label = np.zeros((NonTarget.shape[0], 1), dtype=np.int)

    return Target, Target_label, NonTarget, NonTarget_label


######################################################################
def test_data_extraction_from_mat(Subject):
    path_mat = sio.loadmat('data/' + 'Subject_' + Subject + '_' + 'Test' + '.mat')
    Flashings = path_mat['Flashing'];
    Signals = path_mat['Signal'];
    StimulusCodes = path_mat['StimulusCode']
    if Subject == 'A':
        TargetChars = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'
    if Subject == 'B':
        TargetChars = 'MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'
    return Flashings, Signals, StimulusCodes, TargetChars


######################################################################
def test_data_and_label(Subject, samps):
    Flashings, Signals, StimulusCodes, TargetChars = test_data_extraction_from_mat(Subject)
    Target = [];
    NonTarget = []
    for i in range(len(Flashings)):
        Flashing = Flashings[i];
        Signal = Signals[i];
        StimulusCode = StimulusCodes[i];
        TargetChar = TargetChars[i]
        Signal = bandpass(Signal, [0.1, 20.0], 240)
        loactions = ERP_location(Flashing)
        sgl_Target = Single_trial_target(TargetChar)
        loca_dicts = Single_local_information(loactions, StimulusCode)
        Target_ERPs, NonTarget_ERPs = Single_trial_ERP(Signal, sgl_Target, loca_dicts, samps)
        Target += [Target_ERPs];
        NonTarget += [NonTarget_ERPs]
    Target = np.array(Target);
    NonTarget = np.array(NonTarget)
    Target = Target.reshape(-1, samps, 64)
    NonTarget = NonTarget.reshape(-1, samps, 64)
    Target_label = np.ones((Target.shape[0], 1), dtype=np.int)
    NonTarget_label = np.zeros((NonTarget.shape[0], 1), dtype=np.int)
    return Target, Target_label, NonTarget, NonTarget_label