VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} OutputResultForm 
   Caption         =   "Gauge Update Status"
   ClientHeight    =   5124
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   7260
   OleObjectBlob   =   "OutputResultForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "OutputResultForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False


Option Explicit

Dim xlApp As Excel.Application
Dim xlWB As Excel.Workbook

Private Sub BrowseButton2_Click()

    ExcelLocationTextBox.Text = BrowseForFolder
    
End Sub

Private Sub CloseButton2_Click()

    Unload OutputResultForm
    
End Sub

Private Sub ExportToExcelButton_Click()

    Dim oFSO As Object
    Dim FolderExists As Boolean

    Set oFSO = CreateObject("Scripting.FileSystemObject")

    If ExcelLocationTextBox.Text = "" Then

        MsgBox "Please select output location!", vbCritical
        Exit Sub
        
    ElseIf OutputResultForm.FileNameTextBox.Text = "" Then
        
        MsgBox "Please Enter the Export File Name!", vbCritical
        Exit Sub

    ElseIf OutputResultForm.StatusListBox.ListCount <= 1 Then

        MsgBox ("List is Empty")
        Exit Sub

    Else
        FolderExists = oFSO.FolderExists(OutputResultForm.ExcelLocationTextBox.Text)
        If FolderExists = False Then

            MsgBox ("Please select Valid output location!")
            Exit Sub

        End If
    End If

    Call CopyListBoxToExcel

On Error GoTo Label1:

    Call SaveExcel

    Call CleanExcel

    Exit Sub

Label1:

    Call CleanExcel
    MsgBox "Macro NOT run completely"

End Sub

Function CopyListBoxToExcel()

    Set xlApp = New Excel.Application
    xlApp.Visible = False

    Set xlWB = xlApp.Workbooks.Add

    Dim rowIndex As Integer
    Dim columnIndex As Integer


    For rowIndex = 0 To OutputResultForm.StatusListBox.ListCount - 1

        For columnIndex = 0 To OutputResultForm.StatusListBox.ColumnCount - 1

        xlWB.Sheets(1).Cells(rowIndex + 1, columnIndex + 1).value = OutputResultForm.StatusListBox.List(rowIndex, columnIndex)

        Next

    Next


End Function

Private Sub SaveExcel()


    Dim StrFileName As String
    StrFileName = GetUniqueXlFileName

    xlWB.SaveAs FileName:=StrFileName, FileFormat:=51

    MsgBox "File Saved Successfully"

End Sub

Private Sub CleanExcel()

    xlApp.Visible = True

    xlWB.Close False
    Set xlWB = Nothing

    Set xlWB = Nothing
    xlApp.Quit

    Set xlApp = Nothing

    Unload OutputResultForm

End Sub

Private Function GetUniqueXlFileName() As String

    GetUniqueXlFileName = OutputResultForm.ExcelLocationTextBox.Text & "\" & OutputResultForm.FileNameTextBox.Text
    Dim i As Integer
    i = 1

    While (Not (Dir(GetUniqueXlFileName & ".xlsx") = ""))

        GetUniqueXlFileName = OutputResultForm.ExcelLocationTextBox.Text & "\" & OutputResultForm.FileNameTextBox.Text & " (" & i & ")"
        i = i + 1
    Wend

End Function

Private Sub UserForm_Initialize()
    
    OutputResultForm.StatusListBox.Clear
    OutputResultForm.StatusListBox.AddItem
    OutputResultForm.StatusListBox.List(0, 0) = "PART NUMBER"
    OutputResultForm.StatusListBox.List(0, 1) = "PROFILE NAME"
    OutputResultForm.StatusListBox.List(0, 2) = "UPDATE STATUS"
    
End Sub
