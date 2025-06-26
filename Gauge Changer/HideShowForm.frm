VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} HideShowForm 
   Caption         =   "Hide/ Show Components"
   ClientHeight    =   5556
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   6144
   OleObjectBlob   =   "HideShowForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "HideShowForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
Private Sub removeCompButton_Click()

    Dim i As Integer
    
    Dim swModel As SldWorks.ModelDoc2
    Set swModel = swApp.ActiveDoc
    
    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swModel.SelectionManager
    
    If swSelect.GetSelectedObjectCount2(-1) > 0 Then
        
        Dim vComps As Variant
        vComps = GetSelectedComponents(swSelect)
        
        For i = LBound(vComps) To UBound(vComps)
            
            Dim swComp As SldWorks.Component2
            Set swComp = vComps(i)
            
            Dim swCompModel As SldWorks.ModelDoc2
            Set swCompModel = swComp.GetModelDoc2()
            
            If (partDict.Exists(swCompModel.GetPathName)) Then
                
                RemoveItemFromList swCompModel.GetPathName
            
            End If
        
        Next i
    
    Else
    
        With Me.StatusListBox
        
            Dim isListItemSelected As Boolean
            isListItemSelected = False
            
            For i = .ListCount - 1 To 0 Step -1
            
                If .Selected(i) = True Then
                    
                    isListItemSelected = True
                    Call HideComponent(.List(i, 2))
                    partDict.Remove .List(i, 2)
                    .RemoveItem (i)

                End If
                
            Next i
            
        End With
            
        If False = isListItemSelected Then

            MsgBox "No items selected to remove from the list", vbInformation
            
        End If
        
    End If
    

End Sub

Private Sub HideComponent(ItemName As String)
    
    Dim Part As PartDoc
    Set Part = partDict.item(ItemName)
    
    Dim swComp As SldWorks.Component2
    Set swComp = Part.GetComponent
    
    swComp.Visible = False
    

End Sub

Function GetSelectedComponents(swSelectionMgr As SldWorks.SelectionMgr) As Variant

    Dim compDict As Scripting.Dictionary
    Set compDict = New Scripting.Dictionary

    Dim i As Integer
    For i = 0 To swSelectionMgr.GetSelectedObjectCount2(-1) - 1
            
        Dim swComp As SldWorks.Component2
        Set swComp = swSelectionMgr.GetSelectedObjectsComponent4(i + 1, -1)
            
        If False = compDict.Exists(swComp.Name2) Then
                
            compDict.Add swComp.Name2, swComp
            
        End If

    Next i
        
    
    If Not (compDict.Count = 0) Then
    
        GetSelectedComponents = compDict.Items
        
    End If

End Function

Private Function RemoveItemFromList(ItemName As String)
    
    Dim i As Integer
    With Me.StatusListBox

        For i = .ListCount - 1 To 0 Step -1
                
            If .List(i, 2) = ItemName Then
            
                Call HideComponent(ItemName)
                partDict.Remove ItemName
                .RemoveItem (i)

                        
            End If
                    
        Next i
        
    End With
    
End Function
Private Sub UpdateGaugeButton_Click()
    
    Dim Gauge As String
    Gauge = UpdateGaugeForm.GaugeComboBox.value
    
    Unload UpdateGaugeForm

    If partDict.Count > 0 Then
                
        Call UpdateGauge(partDict, Gauge)
        Unload Me
        OutputResultForm.Show vbModeless
    
    End If


    
End Sub

Private Sub UpdateGauge(partDict As Scripting.Dictionary, Gauge As String)
    
    Dim vItems As Variant
    vItems = partDict.Items
    
    Dim IsInit As Boolean
    IsInit = True
    
    Dim VaultName As String
    Dim pdmVault As EdmVault5
    
    Dim i As Integer
    For i = LBound(vItems) To UBound(vItems)
    
        Me.ProgressLabel.Caption = "Updating " & i + 1 & " of " & UBound(vItems) + 1
        Me.Repaint
        
        Dim newPart As PartDoc
        Set newPart = vItems(i)

        Dim swDoc As SldWorks.ModelDoc2
        Set swDoc = newPart.GetModelDocObject
            
        Set swDoc = swApp.ActivateDoc3(swDoc.GetPathName, True, swRebuildOnActivation_e.swDontRebuildActiveDoc, Err)
                    
        If newPart.IsSheetMetal Then
                    
            Dim newPartSheetMetal As ISheetMetal
            Set newPartSheetMetal = NewISheetMetal(newPart)
        
            newPartSheetMetal.UpdateThickness Gauge
    
            Dim lErrors As Long
            Dim longWarnings As Long
            Dim boolstatus As Boolean
            boolstatus = swDoc.Save3(swSaveAsOptions_Silent, lErrors, longWarnings)
                
            If IsInit Then
                
                VaultName = LoginAndGetVaultName(swDoc, pdmVault)
                IsInit = False
                    
            End If
                   
            Dim IsCheckedOut As Boolean
            IsCheckedOut = CheckWhetherThisPartIsLocked(VaultName, pdmVault, swDoc)
    
            Dim PNo As String
            PNo = newPart.GetPartNumber
                
            Dim Profile As String
            Profile = newPart.Profile
                
            If newPartSheetMetal.Gauge = Gauge Then
                
                Call AddtoListBox(OutputResultForm.StatusListBox, PNo, Profile, GetRemarks("Gauge Changed Successfully.", IsCheckedOut))
                    
            Else
                
                Call AddtoListBox(OutputResultForm.StatusListBox, PNo, Profile, GetRemarks("Gauge Not Changed Successfully for some reason.", IsCheckedOut))
                    
            End If
                
            If Not swDoc Is Nothing Then
                
                swApp.CloseDoc swDoc.GetPathName
                    
            End If
    
        End If
    
    Next i
    
    Unload Me
    
End Sub


Function GetRemarks(Remarks As String, IsCheckedOut As Boolean) As String

    If False = IsCheckedOut Then
        
        GetRemarks = Remarks & " But Part is not checked out!!!"
        
    End If
    
End Function


Function LoginAndGetVaultName(swPart As SldWorks.ModelDoc2, ByRef pdmVault As EdmVault5) As String

    Set pdmVault = New EdmVault5
        
    On Error GoTo Label1
        
        Dim VaultName As String
        VaultName = pdmVault.GetVaultNameFromPath(swPart.GetPathName)
        
        If Not VaultName = "" Then
        
            pdmVault.LoginAuto VaultName, 0
            LoginAndGetVaultName = VaultName
        
        End If

Label1:
        If Err Then
            
            Err.Clear
            
        End If

    
End Function

Function CheckWhetherThisPartIsLocked(VaultName As String, pdmVault As EdmVault5, swPart As SldWorks.ModelDoc2) As Boolean

    CheckWhetherThisPartIsLocked = False

    If Not VaultName = "" Then

        If pdmVault.IsLoggedIn Then
                
            Dim userMgr As IEdmUserMgr5
            Set userMgr = pdmVault
                    
            Dim user As IEdmUser5
            Set user = userMgr.GetLoggedInUser
                    
            Dim parentFolder As IEdmFolder5
            Dim pdmFile As IEdmFile5
            Set pdmFile = pdmVault.GetFileFromPath(swPart.GetPathName, parentFolder)
                    
            If Not pdmFile Is Nothing Then
                    
                If pdmFile.IsLocked Then
                        
                    If pdmFile.LockedByUser.Name = user.Name Then
                                    
                        CheckWhetherThisPartIsLocked = True
                                
                    Else
                                
                        CheckWhetherThisPartIsLocked = False
                        'MsgBox "Some Files are not checked out by the " & user.Name
                        Exit Function
                                
                    End If
                            

                End If
                        
            End If
                    
        End If
                
    Else
                
        CheckWhetherThisPartIsLocked = True
    
    End If

End Function

Private Sub UserForm_Initialize()
    
    Me.StatusListBox.Clear
    Me.StatusListBox.AddItem
    Me.StatusListBox.List(0, 0) = "PART NUMBER"
    Me.StatusListBox.List(0, 1) = "PROFILE NAME"
    'Me.StatusListBox.List(0, 2) = "GAUGE"
    
End Sub

'Private Sub UpdateGauge(CheckedOutPathList As Variant, AssyPathName As String, Gauge As String)
'
'    Dim swDocSpecification As SldWorks.DocumentSpecification
'    Set swDocSpecification = swApp.GetOpenDocSpec(AssyPathName)
'
'    swDocSpecification.DocumentType = swDocumentTypes_e.swDocASSEMBLY
'    swDocSpecification.ReadOnly = False
'    swDocSpecification.Silent = True
'
'    Dim swAssy As SldWorks.ModelDoc2
'    Set swAssy = swApp.OpenDoc7(swDocSpecification)
'
'    Dim i As Integer
'    For i = LBound(CheckedOutPathList) To UBound(CheckedOutPathList)
'
'        Me.ProgressLabel.Caption = "Updating " & i + 1 & " of " & UBound(CheckedOutPathList) + 1
'
'        Dim oldPart As PartDoc
'        Set oldPart = partDict.item(CheckedOutPathList(i))
'
'        If profileDict.Exists(oldPart.Profile) Then
'
'            Dim swDoc As IModelDoc2
'            Set swDoc = swApp.ActivateDoc3(CheckedOutPathList(i), True, swRebuildOnActivation_e.swDontRebuildActiveDoc, Err)
'
'            If Not swDoc Is Nothing Then
'
'                Dim newPart As PartDoc
'                Set newPart = NewPartDoc(swDoc, swDoc.ConfigurationManager.ActiveConfiguration.Name)
'
'                If newPart.IsSheetMetal Then
'
'                    Dim newPartSheetMetal As ISheetMetal
'                    Set newPartSheetMetal = NewISheetMetal(newPart)
'
'                    newPartSheetMetal.UpdateThickness Gauge
'
'                    swDoc.ForceRebuild3 False
'
'                    Dim lErrors As Long
'                    Dim longWarnings As Long
'                    Dim boolstatus As Boolean
'                    boolstatus = swDoc.Save3(swSaveAsOptions_Silent, lErrors, longWarnings)
'
'
'                End If
'
'                swApp.CloseDoc swDoc.GetPathName
'
'
'            End If
'
'        End If
'
'    Next i
'
'    Unload Me
'
'End Sub

