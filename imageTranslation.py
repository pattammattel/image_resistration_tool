from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout, QFileDialog, QMenuBar, QMenu, QLabel, QDoubleSpinBox, QPushButton
from PyQt5 import QtCore, QtGui
from numpy import NaN
from pandas import *
import sys, csv

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.window = QMainWindow(self)
        self.window_width, self.window_height = 1500, 600
        self.resize(self.window_width, self.window_height)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.table = TableView(data, 17, 14)
        self.table.verticalHeader().hide()
        layout.addWidget(self.table)
        self.table.viewport().installEventFilter(self)

        menuBar = QMenuBar(self)
        layout.addWidget(menuBar, 0)
        fileMenu = menuBar.addMenu("File")

        impMenu = QMenu('Import', self)
        expMenu = QMenu('Export', self)
        impAct = QAction('Import Excel File', self)
        impAct2 = QAction('Import CSV File', self)
        expAct = QAction('Export Excel File', self)
        expAct2 = QAction('Export CSV File', self)

        impAct.triggered.connect(lambda: (loadExcelFile(self)))
        impAct2.triggered.connect(lambda: (loadCSVFile(self)))
        expAct.triggered.connect(lambda: (writeExcelFile(self)))
        expAct2.triggered.connect(lambda: (writeCSVFile(self)))
        impMenu.addAction(impAct)
        impMenu.addAction(impAct2)
        expMenu.addAction(expAct)
        expMenu.addAction(expAct2)

        fileMenu.addMenu(impMenu)

        fileMenu.addMenu(expMenu)

        self.label = QLabel()
        self.label.setText("Selected Row: ")
        layout.addWidget(self.label)
        self.spinBox = QDoubleSpinBox()
        self.spinBox.clear()
        self.spinBox.setRange(self.table.rowCount()-(self.table.rowCount()-1) if self.table.rowCount()-(self.table.rowCount()-1) > 0 else "", 
        self.table.rowCount() if self.table.rowCount() > 0 else "")
        self.spinBox.setDecimals(0)
        layout.addWidget(self.spinBox)
        self.button = QPushButton()
        self.button.setText('Export')
        layout.addWidget(self.button)
        self.button.clicked.connect(lambda: (self.export_data()))
        self.table.cellClicked.connect(lambda: (self.on_selection_changed()))
        self.spinBox.valueChanged.connect(lambda: (self.on_spin_changed()))

    
    def on_selection_changed(self):
        self.spinBox.setValue(self.table.currentRow())

    def on_spin_changed(self):
        self.table.selectRow(int(self.spinBox.value()))
        
    def export_data(self):   
        for i in range(13):
            print(self.table.item(self.table.currentRow(), i).text(), end=' ')
        print(self.table.item(self.table.currentRow(), 13).text())
            

data = {'Point #': ['Origin','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'],
        'HXN Rel. Coordinates X[um]':['0','-37','1190','1120','33','324','245','47','41','204','6','0','777','108','98','74','65'],
        'HXN Rel. Coordinates Y[um]':['0','-375','-611','2308','2519','1849','698','707','506','196','202','0','1254','652','549','186','82'],
        'Nikon Coordinates X[mm]':['61.829','61.866','60.639','60.709','61.796','61.505','61.584','61.782','61.788','61.625','61.823','61.829','61.052','61.721','61.731','61.755','61.764'],
        'Nikon Coordinates Y[mm]':['30.72','31.095','31.331','28.412','28.201','28.871','30.022','30.013','30.214','30.524','30.518','30.72','29.466','30.068','30.171','30.534', '30.638'],
        'Relative Coordinates X[mm]':[],
        'Relative Coordinates Y[mm]':[],
        'Change Sign X[mm]':[],
        'Change Sign Y[mm]':[],
        'In Microns X[um]': [],
        'In Microns Y[um]': [],
        'Point': ['Origin','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'],
        'HXN Absolute Coordinates X[um]': ['-1168'],
        'HXN Absolute Coordinates Y[um]': ['-90']}
 
class TableView(QTableWidget):
    def __init__(self, data, *args):
        QTableWidget.__init__(self, *args)
        self.data = data
        self.setData()
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def setData(self): 
        horHeaders = ['Point #', 'HXN Rel. Coordinates X[um]', 'HXN Rel. Coordinates Y[um]', 'Nikon Coordinates X[mm]', 'Nikon Coordinates Y[mm]',
         'Relative Coordinates X[mm]', 'Relative Coordinates Y[mm]','Change Sign X[mm]','Change Sign Y[mm]','In Microns X[um]','In Microns Y[um]',
         'Point #','HXN Absolute Coordinates X[um]', 'HXN Absolute Coordinates Y[um]']
        for n, key in enumerate(self.data.keys()):
            #horHeaders.append(key)
            for m, item in enumerate(self.data[key]):
                newitem = QTableWidgetItem(item)
                self.setItem(m, n, newitem)
        self.setHorizontalHeaderLabels(horHeaders)
        for i in range(17):
            newitem = QTableWidgetItem(str((float(self.item(i,1).text())/1000) * -1))
            self.setItem(i, 5, newitem)
            newitem = QTableWidgetItem(str((float(self.item(i,2).text())/1000) * -1))
            self.setItem(i, 6, newitem)
            newitem = QTableWidgetItem(str((float(self.item(i,1).text())/1000)))
            self.setItem(i, 7, newitem)
            newitem = QTableWidgetItem(str((float(self.item(i,2).text())/1000)))
            self.setItem(i, 8, newitem)
            newitem = QTableWidgetItem(self.item(i,1).text())
            self.setItem(i, 9, newitem)
            newitem = QTableWidgetItem(self.item(i,2).text())
            self.setItem(i, 10, newitem)
        for i in range(1,17):
            newitem = QTableWidgetItem(str(float(self.item(i,1).text()) + float(self.item(0, 12).text())))
            self.setItem(i, 12, newitem)
            newitem = QTableWidgetItem(str(float(self.item(i,2).text()) + float(self.item(0, 13).text())))
            self.setItem(i, 13, newitem)
        self.itemChanged.connect(self.valueChangedX)
        self.itemChanged.connect(self.valueChangedY)
        

    def valueChangedX(self, no):
        if no.column()==1:
            newitem = QTableWidgetItem(str((float(self.item(no.row(),1).text())/1000) * -1))
            self.setItem(no.row(), 5, newitem)
            newitem = QTableWidgetItem(str((float(self.item(no.row(),1).text())/1000)))
            self.setItem(no.row(), 7, newitem)
            newitem = QTableWidgetItem(self.item(no.row(),1).text())
            self.setItem(no.row(), 9, newitem)
            newitem = QTableWidgetItem(str(float(self.item(no.row(),1).text()) + float(self.item(0, 12).text())))
            self.setItem(no.row(), 12, newitem)
        if no.column()==12 and no.row()==0:
            for i in range(1,17):
                newitem = QTableWidgetItem(str(float(self.item(i,1).text()) + float(self.item(0, 12).text())))
                self.setItem(i, 12, newitem)


    def valueChangedY(self, no):
        if no.column()==2:
            newitem = QTableWidgetItem(str((float(self.item(no.row(),2).text())/1000) * -1))
            self.setItem(no.row(), 6, newitem)
            newitem = QTableWidgetItem(str((float(self.item(no.row(),2).text())/1000)))
            self.setItem(no.row(), 8, newitem)
            newitem = QTableWidgetItem(self.item(no.row(),2).text())
            self.setItem(no.row(), 10, newitem)
            newitem = QTableWidgetItem(str(float(self.item(no.row(),2).text()) + float(self.item(0, 13).text())))
            self.setItem(no.row(), 13, newitem)
        if no.column()==13 and no.row()==0:
            for i in range(1,17):
                newitem = QTableWidgetItem(str(float(self.item(i,2).text()) + float(self.item(0, 13).text())))
                self.setItem(i, 13, newitem)

def loadCSVFile(self):
    self.file_name = QFileDialog().getOpenFileName(self, "Select .csv File", '', '.csv file(*csv )')
    if self.file_name[0]:
        df = read_csv(self.file_name[0])
        col = {'Point #', 'HXN Rel. Coordinates X[um]', 'HXN Rel. Coordinates Y[um]', 'Nikon Coordinates X[mm]',
        'Nikon Coordinates Y[mm]', 'Relative Coordinates X[mm]', 'Relative Coordinates Y[mm]', 'Change Sign X[mm]',
        'Change Sign Y[mm]', 'In Microns X[um]', 'In Microns Y[um]', 'Point #.1', 'HXN Absolute Coordinates X[um]', 
        'HXN Absolute Coordinates Y[um]'}
        for k in col:
            if k == 0 or k == 11:
                for i, item in enumerate(df[col[k]]):
                    if isinstance(item, str) == False:
                        print(i, str(item))
                        newitem = QTableWidgetItem(str(int(item)))
                        self.table.setItem(i, k, newitem)
                    elif item == "":
                        newitem = QTableWidgetItem(str(int(0)))
                        self.table.setItem(i, k, newitem)
                    else:
                        newitem = QTableWidgetItem(str(item))
                        self.table.setItem(i, k, newitem)
            else:
                for i, item in enumerate(df[col[k]]):
                    if isinstance(item, str) == False:
                        print(i, str(item))
                        newitem = QTableWidgetItem(str(round(float(item), 3)))
                        self.table.setItem(i, k, newitem)
                    elif item == "":
                        newitem = QTableWidgetItem(str(round(float(0), 3)))
                        self.table.setItem(i, k, newitem)
                    else:
                        newitem = QTableWidgetItem(str(item))
                        self.table.setItem(i, k, newitem)
           

         

def writeCSVFile(self):
    path, ok = QFileDialog.getSaveFileName(self, 'Save CSV', '', 'CSV(*.csv)')
    if ok:
        columns = range(self.table.columnCount())
        header = [self.table.horizontalHeaderItem(column).text()
                    for column in columns]
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', lineterminator='\n')
            writer.writerow(header)
            for row in range(self.table.rowCount()):
                writer.writerow(
                    self.table.item(row, column).text()
                    for column in columns)


def writeExcelFile(self):
    path, ok = QFileDialog.getSaveFileName(self, 'Save Excel', '', 'excel file(*xsls *xslx)')
    if ok:
        columnHeaders = []
        # create column header list
        for j in range(self.table.columnCount()):
            columnHeaders.append(self.table.horizontalHeaderItem(j).text())

        df = DataFrame(columns=columnHeaders)

        # create dataframe object recordset
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                df.at[row, columnHeaders[col]] = self.table.item(row, col).text()

        df.to_excel(path, index=False)
        print('Excel file exported')

    
def loadExcelFile(self):
    self.file_name = QFileDialog().getOpenFileName(self, "Select Excel File", '',
                                                                 'excel file(*xsls *xlsx )')
                                                                 
    if self.file_name[0]:
        xls = ExcelFile(self.file_name[0])
        df = xls.parse(xls.sheet_names[0])
        col = {'Point #', 'HXN Rel. Coordinates X[um]', 'HXN Rel. Coordinates Y[um]', 'Nikon Coordinates X[mm]',
        'Nikon Coordinates Y[mm]', 'Relative Coordinates X[mm]', 'Relative Coordinates Y[mm]', 'Change Sign X[mm]',
        'Change Sign Y[mm]', 'In Microns X[um]', 'In Microns Y[um]', 'Point #.1', 'HXN Absolute Coordinates X[um]', 
        'HXN Absolute Coordinates Y[um]'}

        for k in col:
            if k == 0 or k == 11:
                for i, item in enumerate(df[col[k]]):
                    if isinstance(item, str) == False:
                        print(i, str(item))
                        newitem = QTableWidgetItem(str(int(item)))
                        self.table.setItem(i, k, newitem)
                    elif item == "":
                        newitem = QTableWidgetItem(str(int(0)))
                        self.table.setItem(i, k, newitem)
                    else:
                        newitem = QTableWidgetItem(str(item))
                        self.table.setItem(i, k, newitem)
            else:
                for i, item in enumerate(df[col[k]]):
                    if isinstance(item, str) == False:
                        print(i, str(item))
                        newitem = QTableWidgetItem(str(round(float(item), 3)))
                        self.table.setItem(i, k, newitem)
                    elif item == "":
                        newitem = QTableWidgetItem(str(round(float(0), 3)))
                        self.table.setItem(i, k, newitem)
                    else:
                        newitem = QTableWidgetItem(str(item))
                        self.table.setItem(i, k, newitem)
 
def eventFilter(self, source, event):
    print('yep')
    if event.type() == QtCore.QEvent.MouseButtonPress:
        if source.column() == 0:
            index = self.table.indexAt(event.pos())
            if index.data():
                print(index.data())
    #return super().eventFilter(source, event)


 
def main(args):
    app = QApplication(args)
    app.setStyleSheet('''
        QWidget {
            font-size: 17px;
        }
    ''')
    myApp = MyApp()
    myApp.show()
    
    sys.exit(app.exec_())
 
if __name__=="__main__":
    main(sys.argv)