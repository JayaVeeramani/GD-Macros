VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} UpdateGaugeForm 
   Caption         =   "Gauge Changer"
   ClientHeight    =   6156
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   5688
   OleObjectBlob   =   "UpdateGaugeForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "UpdateGaugeForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
Option Explicit


Const Macro_Name As String = "UPDATE TO ROLL FORMER PANELS"
Const VAULT_NAME As String = "FBD"

Dim pdmVault As EdmVault5

Dim swApp As SldWorks.SldWorks
Dim swModel As SldWorks.ModelDoc2
Dim swAssyDoc As AssemblyDoc

Dim partDict As Scripting.Dictionary
Dim profileDict As Scripting.Dictionary
Dim IsIsolateClicked As Boolean

Private Sub AddProfileButton_Click()

    With UpdateGaugeForm
    
        If .availableProfilesComboBox.value = "" Then
        
            MsgBox "Please select the profile from combo box and then click the button"
            
        Else
            
            If ProfileAlreadyExists(.availableProfilesComboBox.value) Then
                
                MsgBox "Profile cannot be added. Selected Profile Already Exists in the list"
            
            Else
                .profileToIsolateList.AddItem .availableProfilesComboBox.value
                
            End If
        
        End If
    
    End With

End Sub


Private Sub IsolateProfileButton_Click()

    IsIsolateClicked = True
    Call IsolateProfilesInList

End Sub

Private Sub RemoveProfileButton_Click()

    Dim i As Integer
    
    With UpdateGaugeForm.profileToIsolateList
    
        For i = .ListCount - 1 To 0 Step -1
        
            If .Selected(i) = True Then
            
                .RemoveItem (i)
                
            End If
            
        Next
        
    End With
End Sub

Private Sub IsolateProfilesInList()
    
    If UpdateGaugeForm.profileToIsolateList.ListCount = 0 Then
    
        MsgBox "List is empty. Please add profile to list to isolate"
        
    Else
    
        swModel.Extension.RunCommand swCommands_e.swCommands_Comp_Isolate_Exit, "Exit Isolate"
        swModel.ClearSelection2 True

        Set profileDict = GetProfileToIsolateFromList
        
        Set partDict = swAssyDoc.GetUniquePartDoc
        
        Dim key As Variant
        For Each key In partDict.Keys
    
            Dim Part As PartDoc
            Set Part = partDict(key)
            
            If profileDict.Exists(Part.Profile) Then
                
                Part.GetComponent.Select4 True, Nothing, False
                
            End If
            
        Next
        
        swModel.Extension.RunCommand swCommands_e.swCommands_Comp_Isolate, "Isolate Components With Selected Profile"
        
    End If

End Sub


Private Sub UpdateGaugeButton_Click()
    
    Dim Gauge As String
    Gauge = UpdateGaugeForm.GaugeComboBox.value
    
    If Gauge = "" Then
    
        MsgBox "No Gauge Value is selected. Please select the Gauge Value"
        
    Else
    
        If IsIsolateClicked Then
            
            Set partDict = swAssyDoc.GetUniquePartDoc
            
        Else
        
            Call IsolateProfilesInList
            Set partDict = swAssyDoc.GetUniquePartDoc
            
        End If

                
        Dim CheckedOutStdPartsList As Variant
        CheckedOutStdPartsList = CheckOutStdPart(partDict.Keys)
    
        If Not IsEmpty(CheckedOutStdPartsList) Then
                
            Call UpdateGauge(CheckedOutStdPartsList, swModel.GetPathName, Gauge)
    
        End If
        
    End If

End Sub

Private Sub UserForm_Initialize()

    Set swApp = Application.SldWorks
    Set swModel = swApp.ActiveDoc
    Set swAssyDoc = NewAssemblyDoc(swModel)
            
    With UpdateGaugeForm
        
        .GaugeComboBox.AddItem "12GA"
        .GaugeComboBox.AddItem "14GA"
        .GaugeComboBox.AddItem "16GA"
        .GaugeComboBox.AddItem "18GA"
        
        .profileToIsolateList.AddItem "EXT-WALL-1"
        .profileToIsolateList.AddItem "EXT-WALL-2"

    End With

End Sub

Private Function ProfileAlreadyExists(ProfileName As String) As Boolean

    ProfileAlreadyExists = False
    Dim i As Integer
    
    With UpdateGaugeForm.profileToIsolateList
    
        For i = 0 To .ListCount - 1
        
            If .List(i, 0) = ProfileName Then
            
                ProfileAlreadyExists = True
                Exit Function
                
            End If
            
        Next
        
    End With
    
End Function

Private Function GetProfileToIsolateFromList() As Object

    Set GetProfileToIsolateFromList = CreateObject("Scripting.Dictionary")
    
    Dim i As Integer
    With UpdateGaugeForm.profileToIsolateList
    
        For i = 0 To .ListCount - 1
        
          GetProfileToIsolateFromList.Add .List(i, 0), .List(i, 0)
            
        Next
        
    End With
    
End Function

Function CheckOutStdPart(PathList) As Variant

    Dim checkedOutList As New IArrList
    
    Dim pdmVault As EdmVault5
    Set pdmVault = New EdmVault5
                
On Error Resume Next
        
    Dim VaultName As String
    VaultName = pdmVault.GetVaultNameFromPath(PathList(0))
        
    If Not VaultName = "" Then
    
        Dim AssyPathName As String
        AssyPathName = swModel.GetPathName
        swApp.CloseDoc AssyPathName
        
        Call CloseStdPartWindows(partDict.Keys)
            
        If Not pdmVault.IsLoggedIn Then
                
            pdmVault.LoginAuto VaultName, 0
                
        End If
    
        Set pdmVault = New EdmVault5
        pdmVault.LoginAuto VaultName, 0
    
        If pdmVault.IsLoggedIn Then
        
            Dim userMgr As IEdmUserMgr5
            Set userMgr = pdmVault
            
            Dim user As IEdmUser5
            Set user = userMgr.GetLoggedInUser
            
            Dim i As Integer
            For i = LBound(PathList) To UBound(PathList)
    
                Dim pdmFile As IEdmFile5
                Dim parentFolder As IEdmFolder5
                Set pdmFile = pdmVault.GetFileFromPath(PathList(i), parentFolder)
                        
                If pdmFile.IsLocked Then
                    If pdmFile.LockedByUser.Name = user.Name Then
                        
                        checkedOutList.AddtoList PathList(i)
                    
                    Else
                    
                        MsgBox PathList(i) & " is already checked out by " & user.Name & ". Can't Copy and Replace this file"
                    
                    End If
                    
                Else
                
                    pdmFile.LockFile parentFolder.ID, 0
                    If pdmFile.IsLocked Then
    
                        checkedOutList.AddtoList PathList(i)
    
                    Else
    
                        MsgBox PathList(i) & " cannot be checked out for some unknown reason." & vbCrLf & " Check whether the part is opened in any solidworks window"
    
                    End If
                
                End If
                
            Next i
        
        End If
        
        CheckOutStdPart = checkedOutList.Items
        
    Else
        
        CheckOutStdPart = PathList
        
    End If
 
End Function

Sub CloseStdPartWindows(PathList As Variant)

    Dim swFrame As SldWorks.Frame
    Set swFrame = swApp.Frame
     
    Dim vDocsWin As Variant
    vDocsWin = swFrame.ModelWindows

    Dim i As Integer
    
    If Not IsEmpty(vDocsWin) Then
    
        For i = LBound(vDocsWin) To UBound(vDocsWin)
        
            Dim swDocWin As SldWorks.ModelWindow
            Set swDocWin = vDocsWin(i)
    
            Dim swRefDoc As SldWorks.ModelDoc2
            Set swRefDoc = swDocWin.ModelDoc
            
            Dim WindowTitle As String
                
            If swRefDoc.GetType = swDocumentTypes_e.swDocPART Then      'creates drawings only for part even if assy is opened in the model window.
                    
'                If Not InStrRev(swDocWin.title, ".") = 0 Then
'                    WindowTitle = Left(swDocWin.title, InStrRev(swDocWin.title, ".") - 1)
'                Else
'                    WindowTitle = swDocWin.title
'                End If
                
                
                If IsExistInStdPartList(swDocWin.title, PathList) Then
                    
                    swApp.CloseDoc swRefDoc.GetPathName
                    
                End If
                
            End If
            
        Next i
        
    End If
End Sub

Private Function IsExistInStdPartList(WindowTitle As String, PathList As Variant) As Boolean
    
    IsExistInStdPartList = False
    
    Dim i As Integer
    For i = LBound(PathList) To UBound(PathList)

        If InStr(UCase(PathList(i)), WindowTitle) > 0 Then
        
            IsExistInStdPartList = True
            Exit Function
            
        End If
        
    Next i

End Function


Private Sub UpdateGauge(CheckedOutPathList As Variant, AssyPathName As String, Gauge As String)
    
    Dim swDocSpecification As SldWorks.DocumentSpecification
    Set swDocSpecification = swApp.GetOpenDocSpec(AssyPathName)

    swDocSpecification.DocumentType = swDocumentTypes_e.swDocASSEMBLY
    swDocSpecification.ReadOnly = False
    swDocSpecification.Silent = True
    
    Dim swAssy As SldWorks.ModelDoc2
    Set swAssy = swApp.OpenDoc7(swDocSpecification)
    
    Dim i As Integer
    For i = LBound(CheckedOutPathList) To UBound(CheckedOutPathList)
    
        Me.ProgressLabel.Caption = "Updating " & i + 1 & " of " & UBound(CheckedOutPathList) + 1
        
        Dim oldPart As PartDoc
        Set oldPart = partDict.item(CheckedOutPathList(i))
        
        If profileDict.Exists(oldPart.Profile) Then

            Dim swDoc As IModelDoc2
            Set swDoc = swApp.ActivateDoc3(CheckedOutPathList(i), True, swRebuildOnActivation_e.swDontRebuildActiveDoc, Err)
            
            If Not swDoc Is Nothing Then
                
                Dim newPart As PartDoc
                Set newPart = NewPartDoc(swDoc, swDoc.ConfigurationManager.ActiveConfiguration.Name)
                
                If newPart.IsSheetMetal Then
                
                    Dim newPartSheetMetal As ISheetMetal
                    Set newPartSheetMetal = NewISheetMetal(newPart)
    
                    newPartSheetMetal.UpdateThickness Gauge
                    
                    swDoc.ForceRebuild3 False
                
                    Dim lErrors As Long
                    Dim longWarnings As Long
                    Dim boolstatus As Boolean
                    boolstatus = swDoc.Save3(swSaveAsOptions_Silent, lErrors, longWarnings)
                
    
                End If
                
                swApp.CloseDoc swDoc.GetPathName
    
    
            End If
            
        End If
    
    Next i
    
    Unload Me
    
End Sub
