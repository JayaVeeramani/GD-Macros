Attribute VB_Name = "FileFunctions"
Option Explicit

Public Function BrowseForFolder(Optional title As String = "Select Folder") As String

    Dim shellApp As Object

    Set shellApp = CreateObject("Shell.Application")

    Dim folder As Object
    Set folder = shellApp.BrowseForFolder(0, title, 0)

    If Not folder Is Nothing Then
        BrowseForFolder = folder.Self.Path
    End If
End Function
