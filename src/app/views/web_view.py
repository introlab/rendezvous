from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtCore import QUrl

class WebView(QWebEngineView):
    def __init__(self, parent=None):
        super(WebView, self).__init__(parent)
        
        self.page = WebPage()
        self.setPage(self.page)
        self.load(QUrl("https://rtcmulticonnection.herokuapp.com/demos"))

    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            event.accept()

class WebPage(QWebEnginePage):
    def __init__(self, parent=None):
        super(WebPage, self).__init__(parent)
        self.featurePermissionRequested.connect(self.onFeaturePermissionRequested)

    def onFeaturePermissionRequested(self, url, feature):
        if feature in (QWebEnginePage.MediaAudioCapture, 
            QWebEnginePage.MediaVideoCapture, 
            QWebEnginePage.MediaAudioVideoCapture):
            self.setFeaturePermission(url, feature, QWebEnginePage.PermissionGrantedByUser)
        else:
            self.setFeaturePermission(url, feature, QWebEnginePage.PermissionDeniedByUser)
