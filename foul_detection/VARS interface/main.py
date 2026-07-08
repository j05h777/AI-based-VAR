import os
os.environ['QT_MULTIMEDIA_PREFERRED_PLUGINS'] = 'windowsmediafoundation'
# or, on some setups:
# os.environ['QT_MEDIA_BACKEND'] = 'windowsmediafoundation'

import sys
from PyQt5.QtWidgets import QApplication

from interface.video_window import VideoWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    player.showMaximized()
    sys.exit(app.exec_())