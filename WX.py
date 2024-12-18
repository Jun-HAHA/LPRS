import cv2
import streamlit
import numpy
import random
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

# YOLO车牌定位
def PlatePosition(OriImage, Model):
    PredictResult = Model.predict(OriImage)
    if list(PredictResult[0].boxes.cls) == []:
        return 0, 0, "识别失败", 999
    PredictImage = PredictResult[0].plot()
    x = int(list(PredictResult[0].boxes.xywh[0])[0])
    y = int(list(PredictResult[0].boxes.xywh[0])[1])
    w = int(list(PredictResult[0].boxes.xywh[0])[2])
    h = int(list(PredictResult[0].boxes.xywh[0])[3])
    c = ["（蓝色）普通车牌", "（绿色）新能源车牌"][int(PredictResult[0].boxes.cls)]
    s = round(float(PredictResult[0].boxes.conf),3)*100
    PlateImage = cv2.cvtColor(numpy.asarray(PredictImage), cv2.COLOR_RGB2BGR)
    PlateImage = numpy.asarray(PlateImage[y-h//2:y+h//2, x-w//2:x+w//2])
    return PredictImage, PlateImage, c, s

# EasyOCR车牌号识别
def PlateOCR(PlateImage, LPRS_OCR):
    if type(PlateImage) == int:
        return "识别失败", 999
    
    PlateText = LPRS_OCR.ocr(PlateImage, cls=True)
    if PlateText[0] == None:
        return "识别失败", 999
    else:
        return PlateText[0][0][1]
    
# 用户评分计算
def ScoreCount():
    with open("TextLib/score.txt", "r", encoding = 'utf-8') as s:
        ScoreAll = s.readlines()
    AvgScore = round(sum(map(int, ScoreAll)) / len(ScoreAll), 1)
    return len(ScoreAll), AvgScore

# 标签页：车牌识别
def LicenseRecognition(Model, OCR):
    streamlit.header("😎欢迎体验LPRS智慧车牌识别功能")
    uploaded_file = streamlit.file_uploader("👉请上传一张带有车牌的图片", type=["jpg", "jpeg", "png"])
    if streamlit.button("没有图片？点这里试一试"):
        uploaded_file = random.sample(["ImageLib/Try_1.jpg","ImageLib/Try_2.jpg","ImageLib/Try_3.jpg","ImageLib/Try_4.jpg"],1)[0]
    if uploaded_file is not None:
        UploadImage = Image.open(uploaded_file)
        PlatePreData = PlatePosition(UploadImage, Model)
        PlateText = PlateOCR(PlatePreData[1], OCR)
        streamlit.markdown("### 识别结果")
        streamlit.write("车牌号识别结果: {}".format(PlateText[0]))
        if PlateText[1] != 999:
            streamlit.write("车牌号识别置信度: {}%".format(round(PlateText[1],2)*100))
        streamlit.write("车牌类型: {}".format(PlatePreData[2]))
        if PlatePreData[3] != 999:
            streamlit.write("车牌分类置信度: {}%".format(PlatePreData[3]))
        if type(PlatePreData[0]) == int and type(PlatePreData[1]) == int:
            streamlit.image(uploaded_file, caption= "该图片识别失败", channels="RGB")
        else:
            streamlit.image(PlatePreData[1], caption= "车牌切割图", channels="RGB")
            streamlit.image(PlatePreData[0], caption= "车牌识别图", channels="BGR")

# 标签页：系统介绍
def Introduce():
    with open("TextLib/Introduce_system.txt", "r", encoding = 'utf-8') as s:
        SystemText = s.readlines()
    for line in SystemText:
        streamlit.markdown(line)
    with streamlit.expander("展开查看模型性能图"):
        streamlit.image("ImageLib/ModelDisplay_1.png", 
                        caption = "模型训练过程", 
                        channels = "RGB", 
                        use_container_width = True)
        col1, col2 = streamlit.columns(2)
        col1.image("ImageLib/ModelDisplay_4.png", 
                    channels = "RGB", 
                    use_container_width = True)
        col2.image("ImageLib/ModelDisplay_2.png", 
                    channels = "RGB", 
                    use_container_width = True)
        col3, col4 = streamlit.columns(2)
        col3.image("ImageLib/ModelDisplay_3.png",  
                    channels = "RGB", 
                    use_container_width = True)
        col4.image("ImageLib/ModelDisplay_5.png", 
                    channels = "RGB", 
                    use_container_width = True)

# 标签页：程序代码
def CodeDisplay():

    # 页面标题
    streamlit.header("📃LPRS主要代码展示")

    # 获取代码
    def CodeToStr(FileName):
        with open(FileName, "r", encoding = 'utf-8') as Code:
            CodeStr = "".join(Code.readlines())
        return CodeStr

    # 展示代码
    with streamlit.expander("📦数据集处理\tLPRS_Dataset.py"):
        streamlit.code(body = CodeToStr("OtherCode/LPRS_Dataset.py"), language = "python", line_numbers = True)
    with streamlit.expander("⚙️模型训练\tLPRS_ModelTrain.py"):
        streamlit.code(body = CodeToStr("OtherCode/LPRS_ModelTrain.py"), language = "python", line_numbers = True)
    with streamlit.expander("📃模型测试\tLPRS_ModelTest.py"):
        streamlit.code(body = CodeToStr("OtherCode/LPRS_ModelTest.py"), language = "python", line_numbers = True)
    with streamlit.expander("💻模型部署\tLPRS_ModelWeb.py"):
        streamlit.code(body = CodeToStr("WX.py"), language = "python", line_numbers = True)

# 标签页：版本介绍
def VesionUpdate():
    streamlit.header("⌛LPRS版本更新表")
    with open("TextLib/VesionUpdate.txt", "r", encoding='utf-8') as VesionFile:
        VesionText = VesionFile.readlines()
    for line in VesionText:
        streamlit.markdown(line)

# 标签页：用户评分
def UserScore():
    streamlit.header("😊请为我们的作品打个分！")
    score = streamlit.slider("👇滑动红点", 0, 10)
    if streamlit.button("提交"):
        streamlit.write("😄感谢您的评分，我们会继续努力！")
        with open("TextLib/score.txt", "a", encoding = 'utf-8') as s:
            s.write(str(score)+"\n")
        ScoreData = ScoreCount()
        streamlit.write("当前用户评分: {}".format(ScoreData[1]))
        streamlit.write("目前有{}人为我们评分".format(ScoreData[0]))

#主函数
def main():

    # 加载模型
    model = YOLO("LPRS_Model.pt")

    # 加载OCR
    cls_model_dir='paddleModels/whl/cls/ch_ppocr_mobile_v2.0_cls_infer'
    rec_model_dir='paddleModels/whl/rec/ch/ch_PP-OCRv4_rec_infer'
    det_model_dir='paddleModels/whl/det/ch/ch_PP-OCRv4_det_infer'
    LPRS_OCR = PaddleOCR(lang="ch",cls_model_dir=cls_model_dir,rec_model_dir=rec_model_dir,det_model_dir=det_model_dir) 

    # 页面标题信息
    streamlit.title("🚘LPRS - 智慧车牌识别系统")
    streamlit.subheader("当前版本: 3.4")
    streamlit.write("《互联网程序设计》课程设计")

    # 标签页按钮
    Page_LicenseRecognition, Page_Introduce, Page_CodeDisplay, Page_VesionUpdate, Page_UserScore = streamlit.tabs(["车牌识别", "系统介绍","代码展示", "版本更新", "用户评分"])

    # 标签页功能
    with Page_LicenseRecognition:
        LicenseRecognition(model, LPRS_OCR)
    with Page_Introduce:
        Introduce()
    with Page_CodeDisplay:
        CodeDisplay()
    with Page_VesionUpdate:
        VesionUpdate()
    with Page_UserScore:
        UserScore()

if __name__ == "__main__":
    main()
