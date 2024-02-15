import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, \
    QDialog, QTextBrowser, QLineEdit, QColorDialog, QComboBox, QScrollArea
from PyQt5.QtGui import QPixmap, QImage, QColor
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessingApp(QWidget):
    def __init__(self, image_path):
        super().__init__()

        self.setWindowTitle("Resim İşleme Uygulaması")
        self.setGeometry(100, 100, 1200, 600)
        
        self.old_image = cv2.imread(image_path)
        self.old_image = cv2.cvtColor(self.old_image, cv2.COLOR_BGR2RGB)
        self.new_image = np.copy(self.old_image)

        

        self.image_info_label = QLabel()
        self.set_image_info_label(self.image_info_label, self.old_image)

        self.border_width_edit = QLineEdit()
        self.border_color_button = QPushButton("Kenar Rengi Seç")
        self.add_border_button = QPushButton("Kenar Ekle")
        self.border_color = (0, 255, 0)

        self.sobel_button = QPushButton("Sobel Uygula")
        self.sobel_button.clicked.connect(self.apply_sobel)
        self.sobel_button.setStyleSheet("background-color: green; color: white")
        self.sobel_button.setFixedSize(150, 30)

        self.deriche_button = QPushButton("Deriche Kenar")
        self.deriche_button.clicked.connect(self.deriche_image)
        self.deriche_button.setStyleSheet("background-color: green; color : white")
        self.deriche_button.setFixedSize(150, 30)


        self.harris_corner_button = QPushButton("Harris Corner")
        self.harris_corner_button.clicked.connect(self.harris_korner)
        self.harris_corner_button.setStyleSheet("background-color: green; color : white")
        self.harris_corner_button.setFixedSize(150,30)

      
        
        
        
        self.init_ui()

    def init_ui(self):
        self.old_image_label = QLabel()
        self.set_image_label(self.old_image_label, self.old_image)

        self.new_image_label = QLabel()
        self.set_image_label(self.new_image_label, self.new_image)

        self.add_button = QPushButton("Resmi Tersine Çevir")
        self.add_button.clicked.connect(self.process_and_show_image)
        self.add_button.setStyleSheet("background-color: green; color: white")
        self.add_button.setFixedSize(150, 30)

        self.cevir_button = QPushButton("Resmin Rengini Tersine Çevir")
        self.cevir_button.clicked.connect(self.cevir)
        self.cevir_button.setStyleSheet("background-color: green; color: white")
        self.cevir_button.setFixedSize(180, 30)

        self.show_info_button = QPushButton("Bilgi Göster")
        self.show_info_button.clicked.connect(self.show_image_info)
        self.show_info_button.setStyleSheet("background-color: green; color: white")
        self.show_info_button.setFixedSize(150, 30)

        self.adaptive_threshold_button = QPushButton("Adaptive Threshold")
        self.adaptive_threshold_button.clicked.connect(self.adaptive_threshold)
        self.adaptive_threshold_button.setStyleSheet("background-color: green; color: white")
        self.adaptive_threshold_button.setFixedSize(150, 30)

        self.otsu_threshold_button = QPushButton("Otsu Threshold")
        self.otsu_threshold_button.clicked.connect(self.otsu_threshold)
        self.otsu_threshold_button.setStyleSheet("background-color: green; color: white")
        self.otsu_threshold_button.setFixedSize(150, 30)

        self.image_info_label.setStyleSheet("background-color: green; color: white")
        self.image_info_label.setFixedSize(200, 50)

        self.border_width_edit.setPlaceholderText("Kenar Genişliği")
        self.border_width_edit.setFixedSize(150, 30)

        self.border_color_button.clicked.connect(self.select_border_color)
        self.border_color_button.setStyleSheet("background-color: green; color: white")
        self.border_color_button.setFixedSize(150, 30)

        self.add_border_button.clicked.connect(self.add_border)
        self.add_border_button.setStyleSheet("background-color: green; color: white")
        self.add_border_button.setFixedSize(150, 30)

        self.blur_type_combobox = QComboBox()
        self.blur_type_combobox.addItem("Bulanıklaştırma Türü Seç")
        self.blur_type_combobox.addItem("Normal Blur")
        self.blur_type_combobox.addItem("Median Blur")
        self.blur_type_combobox.addItem("Box Filter")
        self.blur_type_combobox.addItem("Bilateral Filter")
        self.blur_type_combobox.addItem("Gaussian Blur")
        self.blur_type_combobox.setStyleSheet("background-color: grey; color: black")
        self.blur_type_combobox.setFixedSize(150, 30)


        self.blur_button = QPushButton("Bulanıklaştır")
        self.blur_button.clicked.connect(self.blur_image)
        self.blur_button.setStyleSheet("background-color: green; color: white")
        self.blur_button.setFixedSize(150, 30)


        



        self.lapacian_button = QPushButton("Lapacian")
        self.lapacian_button.clicked.connect(self.lapacian_image)
        self.lapacian_button.setStyleSheet("background-color: green; color: white")
        self.lapacian_button.setFixedSize(150, 30)

        self.cannykenar_button = QPushButton("Canny Kenar")
        self.cannykenar_button.clicked.connect(self.canny_image)
        self.cannykenar_button.setStyleSheet("background-color: green; color:white")
        self.cannykenar_button.setFixedSize(150, 30)


        self.detect_faces_button = QPushButton("Yüzü Tespit Et")
        self.detect_faces_button.clicked.connect(self.detect_faces)
        self.detect_faces_button.setStyleSheet("background-color: green; color: white")
        self.detect_faces_button.setFixedSize(150, 30)


        self.contour_button = QPushButton("Konturları Çiz")
        self.contour_button.clicked.connect(self.draw_contours)
        self.contour_button.setStyleSheet("background-color: green; color: white")
        self.contour_button.setFixedSize(150, 30)
         

        # self.processing_options_combobox = QComboBox()
        # self.processing_options_combobox.addItems(self.processing_options)
        # self.processing_options_combobox.currentIndexChanged.connect(self.perform_watershed)
        # self.watershed_button.setStyleSheet("background-color: green; color: white")
        # self.processing_options_combobox.setFixedSize(150, 30)

    # Add a button to perform watershed segmentation
        # self.watershed_combobox = QComboBox()
        # self.watershed_combobox.addItem("Original")
        # self.watershed_combobox.addItem("Blurred")
        # self.watershed_combobox.addItem("Gray")
        # self.watershed_combobox.addItem("Threshold")
        # self.watershed_combobox.addItem("Morphology")
        # self.watershed_combobox.addItem("Sure Background")
        # self.watershed_combobox.addItem("Distance Transform")
        # self.watershed_combobox.addItem("Sure Foreground")
        # self.watershed_combobox.addItem("Sure Background")
        # self.watershed_combobox.addItem("Watershed Result")
        # self.watershed_combobox.setStyleSheet("background-color: grey; color: black")
        # self.watershed_combobox.setFixedSize(150, 30)

        self.watershed_button = QPushButton("Watershed")
        self.watershed_button.clicked.connect(self.apply_watershed)
        self.watershed_button.setStyleSheet("background-color: green; color: white")
        self.watershed_button.setFixedSize(150, 30)


     
 
        # Scroll alanı oluştur
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Widget oluştur ve tüm düğmeleri, combobox'ı ve giriş widget'larını içermesi için oluştur
        scroll_content = QWidget()
        scroll_area.setWidget(scroll_content)

        # Scroll alanındaki widget için bir düzen oluştur
        scroll_layout = QVBoxLayout(scroll_content)

        # Resimleri yan yana göstermek için yatay bir düzen oluştur
        images_layout = QHBoxLayout()

        # Resimleri yatay düzene ekle
        images_layout.addWidget(self.old_image_label)
        images_layout.addWidget(self.new_image_label)

        # Yatay düzeni ana dikey düzene ekle
        scroll_layout.addLayout(images_layout)

        # Diğer bileşenleri düzene ekle
        scroll_layout.addWidget(self.add_button)
        scroll_layout.addWidget(self.cevir_button)
        scroll_layout.addWidget(self.show_info_button)
        scroll_layout.addWidget(self.adaptive_threshold_button)
        scroll_layout.addWidget(self.otsu_threshold_button)
        scroll_layout.addWidget(self.image_info_label)
        scroll_layout.addWidget(self.border_width_edit)
        scroll_layout.addWidget(self.border_color_button)
        scroll_layout.addWidget(self.add_border_button)
        scroll_layout.addWidget(self.blur_type_combobox)
        scroll_layout.addWidget(self.blur_button)
        scroll_layout.addWidget(self.sobel_button)
        scroll_layout.addWidget(self.lapacian_button)
        scroll_layout.addWidget(self.cannykenar_button)
        scroll_layout.addWidget(self.deriche_button)
        scroll_layout.addWidget(self.harris_corner_button)
        scroll_layout.addWidget(self.detect_faces_button)
        scroll_layout.addWidget(self.contour_button)
        scroll_layout.addWidget(self.watershed_button)
       
      
        

        # Scroll alanına düzeni ayarla
        scroll_content.setLayout(scroll_layout)

        # Ana pencere için düzeni
        # ayarla
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)

    

    def set_image_label(self, label, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

    def set_image_info_label(self, label, image):
        info_text = f"Veri Tipi: {type(image)}\n" \
                    f"Boyutlar: {image.shape}\n"

        label.setText(info_text)

    def process_and_show_image(self):
        self.new_image = cv2.flip(self.old_image, 1)
        self.set_image_label(self.new_image_label, self.new_image)

    def cevir(self):
        self.new_image = cv2.bitwise_not(self.old_image)
        self.set_image_label(self.new_image_label, self.new_image)

    def show_image_info(self):
        info_dialog = QDialog(self)
        info_dialog.setWindowTitle("Resim Bilgisi")

        text_browser = QTextBrowser(info_dialog)
        text_browser.setPlainText(f"Veri Tipi: {type(self.old_image)}\n"
                                   f"Boyutlar: {self.old_image.shape}\n"
                                   f"İçerik:\n{self.old_image}")

        layout = QVBoxLayout(info_dialog)
        layout.addWidget(text_browser)
        info_dialog.setLayout(layout)

        info_dialog.exec_()

    def adaptive_threshold(self):
        gray_image = cv2.cvtColor(self.old_image, cv2.COLOR_RGB2GRAY)
        block_size = 11
        C = 2
        adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                   block_size, C)

        fig, axes = plt.subplots(1, 2)
        fig.suptitle('Adaptive Threshold')

        axes[0].set_title('Orijinal Resim')
        axes[0].imshow(self.old_image)

        axes[1].set_title('Adaptive Threshold')
        axes[1].imshow(adaptive_threshold, cmap='gray')

        fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.35)
        fig.canvas.manager.toolbar.setVisible(False)

        plt.show()

    def otsu_threshold(self):
        gray_image = cv2.cvtColor(self.old_image, cv2.COLOR_RGB2GRAY)
        _, otsu_threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        fig, axes = plt.subplots(1, 2)
        fig.suptitle('Otsu Threshold')

        axes[0].set_title('Orijinal Resim')
        axes[0].imshow(self.old_image)

        axes[1].set_title('Otsu Threshold')
        axes[1].imshow(otsu_threshold, cmap='gray')

        fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.35)
        fig.canvas.manager.toolbar.setVisible(False)

        plt.show()

    def select_border_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.border_color = color.getRgb()[:3]

    def add_border(self):
        border_width_text = self.border_width_edit.text()
        try:
            border_width = int(border_width_text)
        except ValueError:
            print("Kenar genişliği geçerli bir tamsayı değil.")
            return

        image_with_border = cv2.copyMakeBorder(self.old_image, border_width, border_width, border_width, border_width,
                                               borderType=cv2.BORDER_CONSTANT, value=self.border_color)

        self.new_image = image_with_border
        self.set_image_label(self.new_image_label, self.new_image)

    def blur_image(self):
        blur_type = self.blur_type_combobox.currentText()

        if blur_type == "Normal Blur":
            self.new_image = cv2.blur(self.old_image, (5, 5))
        elif blur_type == "Median Blur":
            self.new_image = cv2.medianBlur(self.old_image, 5)
        elif blur_type == "Box Filter":
            self.new_image = cv2.boxFilter(self.old_image, -1, (5, 5))
        elif blur_type == "Bilateral Filter":
            self.new_image = cv2.bilateralFilter(self.old_image, 9, 75, 75)
        elif blur_type == "Gaussian Blur":
            self.new_image = cv2.GaussianBlur(self.old_image, (5, 5), 0)

        self.set_image_label(self.new_image_label, self.new_image)

    def apply_sobel(self):
        sobel_result = cv2.Sobel(self.old_image, cv2.CV_64F, 1, 1, ksize=3)
        sobel_result = np.uint8(np.absolute(sobel_result))

        fig, axes = plt.subplots(1, 2)
        fig.suptitle('Sobel Uygulama')

        axes[0].set_title('Orijinal Resim')
        axes[0].imshow(self.old_image)

        axes[1].set_title('Sobel Uygulama')
        axes[1].imshow(sobel_result, cmap='gray')

        fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.35)
        fig.canvas.manager.toolbar.setVisible(True)

        plt.show()

    def lapacian_image(self):

        laplacian_result = cv2.Laplacian(self.old_image, cv2.CV_64F)
        laplacian_result = np.uint8(np.absolute(laplacian_result))
        fig, axes = plt.subplots(1, 2)
        fig.suptitle('Laplacian Uygulama')
        axes[0].set_title('Orijinal Resim')
        axes[0].imshow(self.old_image, cmap='gray')

        axes[1].set_title('Laplacian Uygulama')
        axes[1].imshow(laplacian_result, cmap='gray')

        fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.35)
        fig.canvas.manager.toolbar.setVisible(True)
        plt.show()

    def canny_image(self):
        low_threshold = 50
        high_threshold = 150
        canny_result = cv2.Canny(self.old_image, low_threshold, high_threshold, L2gradient=True)
        canny_result = np.uint8(np.absolute(canny_result))
        fig, axes = plt.subplots(1, 2)
        fig.suptitle('Canny Kenar Tespiti')

        axes[0].set_title('Orijinal Resim')
        axes[0].imshow(self.old_image, cmap='gray')

        axes[1].set_title('Canny Kenar Tespiti')
        axes[1].imshow(canny_result, cmap='gray')

        fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.35)
        fig.canvas.manager.toolbar.setVisible(True)

        plt.show()

    def deriche_image(self):
         alpha = 7000
         kernel_size = 25
         

    # Deriche filtresi için kernel oluştur
         kx, ky = cv2.getDerivKernels(1, 1, kernel_size, normalize=True)
         deriche_kernel_x = alpha * kx
         deriche_kernel_y = alpha * ky

    # Görüntüyü Deriche Filtresi ile türevle
         deriche_x = cv2.filter2D(self.new_image, -1, deriche_kernel_x)
         deriche_y = cv2.filter2D(self.new_image, -1, deriche_kernel_y)

    # Kenarları birleştir
         edges = np.sqrt(deriche_x**2 + deriche_y**2)

    # Görselleştirme
         fig,axes=plt.subplots(1,2,figsize=(17,7))
         fig.suptitle('Deriche Kenar Tespiti')
         edges = np.uint8(np.absolute(edges))
         self.old_image = np.uint8(np.absolute(self.old_image))
         axes[0].set_title('Orijinal Resim')
         axes[0].imshow(self.old_image, cmap='gray')

         axes[1].set_title('Deriche Kenar Tespiti')
         axes[1].imshow(edges, cmap='gray')

         fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.35)
         fig.canvas.manager.toolbar.setVisible(True)
         plt.show()

    def harris_korner(self):
    # Harris köşe tespiti için parametreleri ayarla
        corner_quality = 0.04
        min_distance   = 10
        block_size     = 3

    # Harris köşe tespiti uygula
        corners = cv2.cornerHarris(cv2.cvtColor(self.old_image, cv2.COLOR_RGB2GRAY), block_size, 3, corner_quality,min_distance)

    # Köşeleri belirli bir eşik değerinde seç
        corners = cv2.dilate(corners, None)
        self.old_image[corners > 0.01 * corners.max()] = [255,0,0]

    # Yeni resmi göster
        self.set_image_label(self.new_image_label, self.old_image)

    # # Görselleştirmeyi sağla
    #     fig, axes = plt.subplots(1, 2)
    #     fig.suptitle('Harris Corner Detection')

    #     axes[0].set_title('Orijinal Resim')
    #     axes[0].imshow(cv2.cvtColor(self.old_image, cv2.COLOR_RGB2BGR))  # RGB'den BGR'ye dönüştürme işlemi

    #     axes[1].set_title('Harris Corner Detection')
    #     axes[1].imshow(corners, cmap='gray')

    # # Ekstra ayarlamalar
    #     fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.35)
    #     fig.canvas.manager.toolbar.setVisible(True)

    #     plt.show()
    # def resize_image(self, image_path, new_width=800, new_height=600):
    #     # Resmi oku
    #     original_image = cv2.imread(image_path)

    #     # Resmi yeniden boyutlandır
    #     resized_image = cv2.resize(original_image, (new_width, new_height))

    #     return resized_image
   

    def detect_faces(self) :
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(self.old_image, scaleFactor=1.3, minNeighbors=6)

        # Yüzleri dikdörtgen ile işaretle
        for (x, y, w, h) in faces:
            cv2.rectangle(self.old_image, (x, y), (x + w, y + h), (255, 0, 0), 10)

        # Yeni resmi göster
        self.set_image_label(self.new_image_label, self.old_image)


    def draw_contours(self):
        gray_image = cv2.cvtColor(self.old_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Resmi üzerine konturları çiz
        cv2.drawContours(self.new_image, contours, -1, (0, 255, 0), 2)

        # Yeni resmi göster
        self.set_image_label(self.new_image_label, self.new_image)    
        
    # def perform_watershed(self):
    #    self.selected_option = self.watershed_combobox.currentText()

    #    if self.selected_option == "Original":
    #     self.new_image = np.copy(self.old_image)
    #    elif self.selected_option == "Blurred":
    #     self.new_image = cv2.blur(self.old_image, (5, 5))
    #    elif self.selected_option == "Gray":
    #     self.new_image = cv2.cvtColor(self.old_image, cv2.COLOR_RGB2GRAY)
    #    elif self.selected_option == "Threshold":
    #     # Threshold işlemi
    #     gray_image = cv2.cvtColor(self.old_image, cv2.COLOR_RGB2GRAY)
    #     _, threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    #     self.new_image = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
    # # Diğer işlemler...
    #    elif self.selected_option == "Morphology":
    #     # Morfolojik işlemler
    #     kernel = np.ones((5, 5), np.uint8)
    #     gray_image = cv2.cvtColor(self.old_image, cv2.COLOR_RGB2GRAY)
    #     _, threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    #     morph_image = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
    #     self.new_image = cv2.cvtColor(morph_image, cv2.COLOR_GRAY2RGB)
    #    elif self.selected_option == "Sure Background":
    #     # Sure Background işlemi
    #     gray_image = cv2.cvtColor(self.old_image, cv2.COLOR_RGB2GRAY)
    #     _, threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    #     sure_bg = cv2.dilate(threshold, None, iterations=3)
    #     self.new_image = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2RGB)
    #    elif self.selected_option == "Distance Transform":
    #     # Distance Transform işlemi
    #     self.apply_distance_transform()
    #    elif self.selected_option == "Sure Foreground":
    #     # Sure Foreground işlemi
    #     gray_image = cv2.cvtColor(self.old_image, cv2.COLOR_RGB2GRAY)
    #     _, threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    #     dist_transform = cv2.distanceTransform(threshold, cv2.DIST_L2, 5)
    #     _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    #     sure_fg = np.uint8(sure_fg)
    #     sure_fg_colored = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2RGB)
    #     self.new_image = sure_fg_colored

    #     self.set_image_label(self.new_image_label, self.new_image)



    # def apply_distance_transform(self):
    # # Distance Transform işlemi
    #    gray_image = cv2.cvtColor(self.old_image, cv2.COLOR_RGB2GRAY)
    #    _, threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    #    dist_transform = cv2.distanceTransform(threshold, cv2.DIST_L2, 5)
    #    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    #    sure_fg = np.uint8(sure_fg)
    #    sure_fg_colored = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2RGB)
    #    self.new_image = sure_fg_colored


    # def update_display(self):
    #     # Seçilen işleme göre görüntüyü güncelle
    #     self.perform_watershed() 



    # def perform_watershed(self):
    #     # Seçilen işleme göre watershed işlemi yap
    #     self.selected_option = self.processing_options_combobox.currentText()
    #     super().perform_watershed()
    def apply_watershed(self):
        imgBlr = cv2.medianBlur(self.old_image, 31)
        imgGray = cv2.cvtColor(imgBlr, cv2.COLOR_BGR2GRAY)

        ret, imgTH = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        imgOPN = cv2.morphologyEx(imgTH, cv2.MORPH_OPEN, kernel, iterations=7)
        sureBG = cv2.dilate(imgOPN, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(imgOPN, cv2.DIST_L2, 5)
        ret, sureFG = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        sureFG = np.uint8(sureFG)
        unknown = cv2.subtract(sureBG, sureFG)

        ret, markers = cv2.connectedComponents(sureFG, labels=5)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(self.old_image, markers)

        contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        imgCopy = self.old_image.copy()
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(imgCopy, contours, i, (255, 0, 0), 5)

        self.new_image = imgCopy
        self.set_image_label(self.new_image_label, self.new_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    options |= QFileDialog.DontUseNativeDialog
    image_path, _ = QFileDialog.getOpenFileName(None, "Resim Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)

    if not image_path:
        sys.exit(0)

    window = ImageProcessingApp(image_path)
    window.show()
    sys.exit(app.exec_())
