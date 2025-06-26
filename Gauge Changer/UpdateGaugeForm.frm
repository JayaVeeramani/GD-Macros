VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} UpdateGaugeForm 
   Caption         =   "Gauge Changer"
   ClientHeight    =   5892
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

Const Macro_Name As String = "GAUGE CHANGER"
Dim pdmVault As EdmVault5
Dim profileDict As Scripting.Dictionary

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

    swModel.Extension.RunCommand swCommands_e.swCommands_Comp_Isolate_Exit, "Exit Isolate"
    swModel.ClearSelection2 True

    Set profileDict = GetProfileToIsolateFromList
    Set partDict = New Scripting.Dictionary

    Dim key As Variant
    For Each key In compDict.Keys
    
        Dim Part As PartDoc
        Set Part = compDict(key)
            
        If profileDict.Exists(Part.Profile) Then
                
            Part.GetComponent.Select4 True, Nothing, False
            partDict.Add CStr(key), Part
            Call AddtoListBox(HideShowForm.StatusListBox, Part.GetPartNumber, Part.Profile, CStr(key))
                
        End If
            
    Next
        
    swModel.Extension.RunCommand swCommands_e.swCommands_Comp_Isolate, "Isolate Components With Selected Profile"

End Sub


Private Sub UpdateGaugeButton_Click()
    
    Dim Gauge As String
    Gauge = UpdateGaugeForm.GaugeComboBox.value
    
    If Gauge = "" Then
    
        MsgBox "No Gauge Value is selected. Please select the Gauge Value"
        
    Else
    
        If Me.profileToIsolateList.ListCount = 0 Then
    
            MsgBox "List is Empty. Please add profile to list to isolate"
        
        Else
            Me.Hide
            Call IsolateProfilesInList
            HideShowForm.Show vbModeless
            
        End If
        
    End If

End Sub

Private Sub UserForm_Initialize()

'    Set swApp = Application.SldWorks
'    Set swModel = swApp.ActiveDoc
'    Set swAssyDoc = NewAssemblyDoc(swModel)
            
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







