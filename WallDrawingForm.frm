VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} WallDrawingForm 
   Caption         =   "Insert Drawing View"
   ClientHeight    =   3864
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   6228
   OleObjectBlob   =   "WallDrawingForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "WallDrawingForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False




Option Explicit

Private Sub CloseButton_Click()
    
    Unload Me
    
End Sub

Private Sub ActivateDSButton_Click()

    If Me.DisplayList.ListIndex = -1 Then
    
         MsgBox "Please Select a Display state of the respective wall from the list", vbCritical, "Select Display State"
         
    Else
        
        If Me.WallNameComboBox.Value = "" Then
        
            MsgBox "Please select the Wall Name from the list", vbExclamation, "Select Wall Name"

        Else
        
            Me.Hide
            
            Dim DisplayStateName As String
            DisplayStateName = Me.DisplayList.List(Me.DisplayList.ListIndex)
        
            swConfig.ApplyDisplayState DisplayStateName
            
            HideShowForm.Show vbModeless
            
        End If
        
    End If

End Sub

Private Sub UserForm_Initialize()

    With Me.WallNameComboBox
    
        .AddItem "Wall-A"
        .AddItem "Wall-B"
        .AddItem "Wall-C"
        .AddItem "Wall-D"
        .AddItem "Ceiling"
        .AddItem "Roof"
        
    End With

End Sub

