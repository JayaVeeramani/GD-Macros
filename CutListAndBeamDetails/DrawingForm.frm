VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} DrawingForm 
   Caption         =   "Insert Drawing View"
   ClientHeight    =   4092
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   6228
   OleObjectBlob   =   "DrawingForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "DrawingForm"
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
        
        If swFloorWeldment Is Nothing Then
        
            MsgBox "Please select the Floor Weldment", vbExclamation, "Floor Weldment Not Selected!"

        Else
        
            Me.Hide
            
            Dim DisplayStateName As String
            DisplayStateName = Me.DisplayList.List(Me.DisplayList.ListIndex)
        
            swConfig.ApplyDisplayState DisplayStateName
            
            HideShowForm.Show vbModeless
            
        End If
        
    End If

End Sub

Private Sub weldmentSelectionButton_Click()

    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swTopLevelModel.SelectionManager
    
    If swSelect.GetSelectedObjectCount2(-1) = 1 Then

        Set swFloorWeldment = swSelect.GetSelectedObjectsComponent4(1, -1)
        
        If Not swFloorWeldment Is Nothing Then
            
            Dim swFloorModel As SldWorks.ModelDoc2
            Set swFloorModel = ResolveAndGetModelDoc(swFloorWeldment)
            
            If swFloorModel.GetType = swDocumentTypes_e.swDocPART Then
                
                If swFloorModel.IsWeldment Then
            
                    Me.FloorSelectionTextBox.Value = "Selected"
                    Me.FloorSelectionTextBox.BackColor = vbGreen
                    
                Else
                
                    MsgBox "Warning! Select component is not a weldment. Please select the floor weldment part", vbCritical, "Selection Warning!"
                
                End If
                
            Else
            
                MsgBox "Warning! Selected component is not a part. Please select the floor weldment part", vbCritical, "Selection Warning!"
                
            End If
            
        Else
        
            Me.FloorSelectionTextBox.Value = "Not Selected"
            Me.FloorSelectionTextBox.BackColor = vbRed
            
        End If

    ElseIf swSelect.GetSelectedObjectCount2(-1) = 0 Then
        
        MsgBox "Warning! Nothing Selected." & vbCrLf & _
        "Please select Floor Weldment component only", vbCritical, "Selection Warning!"
    
    Else
    
    
        MsgBox "Warning! More than one items are selected." & vbCrLf & _
                "Please select Floor Weldment component only", vbCritical, "Selection Warning!"

    End If
    
End Sub
