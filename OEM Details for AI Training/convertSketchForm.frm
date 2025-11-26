VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} convertSketchForm 
   Caption         =   "OEM Details Extracter"
   ClientHeight    =   4476
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   8136
   OleObjectBlob   =   "convertSketchForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "convertSketchForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False


Option Explicit

Dim compDict As Scripting.Dictionary
Dim oConvertInput As ISketchConvertInput

Private Sub AddCompButton_Click()
    
    Dim swModel As SldWorks.ModelDoc2
    Set swModel = swApp.ActiveDoc
    
    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swModel.SelectionManager
    
    If swSelect.GetSelectedObjectCount2(-1) > 0 Then
    
        Dim i As Integer
        
        For i = 1 To swSelect.GetSelectedObjectCount2(-1)
        
            Dim swComp As SldWorks.Component2
            Set swComp = swSelect.GetSelectedObjectsComponent4(i, -1)

            While Not swComp.GetParent Is Nothing
            
                Debug.Print swComp.GetParent.GetPathName
                Debug.Print swModel.GetPathName
                
                Set swComp = swComp.GetParent
  
            
            Wend
            
            If Not (compDict.Exists(swComp.Name2)) Then
                    
                Me.CompListButton.AddItem
                Me.CompListButton.List(Me.CompListButton.ListCount - 1, 0) = swComp.Name2
                compDict.Add swComp.Name2, swComp
                
            End If

        Next i
        
        Call AddCompToObject
        
    Else

        Me.CompListButton.BackColor = vbRed
        MsgBox "No Components were selected"
    
    End If
    
    swModel.ClearSelection2 True
    
End Sub

Private Function AddCompToObject()

    oConvertInput.vCompArr = compDict.Items
    
    If compDict.Count > 0 Then
        
        Me.CompListButton.BackColor = vbGreen
        
            
            
    Else
            
        Me.CompListButton.BackColor = vbRed
            
    End If
    
End Function

Private Sub AddPointButton_Click()
    
    Dim swModel As SldWorks.ModelDoc2
    Set swModel = swApp.ActiveDoc

    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swModel.SelectionManager
    
    If swSelect.GetSelectedObjectType3(1, -1) = swSelectType_e.swSelEXTSKETCHPOINTS Or _
        swSelect.GetSelectedObjectType3(1, -1) = swSelectType_e.swSelVERTICES Then
        
        Dim swPoint As Object
        Set swPoint = swSelect.GetSelectedObject6(1, -1)
        
        Call AddPointToTextBox(swPoint, swSelect)

    End If
    
    swModel.ClearSelection2 True
    Set swModel = Nothing
    
End Sub

Private Function AddPointToTextBox(swPoint As Object, swSelect As SldWorks.SelectionMgr)

    If Not swPoint Is Nothing Then

        Me.pointTextBox.Value = "Point Selected"
        Me.pointTextBox.BackColor = vbGreen
        
        If swSelect.GetSelectedObjectType3(1, -1) = swSelectType_e.swSelVERTICES Then
        
            Set oConvertInput.swSketchPoint = Nothing
            Set oConvertInput.swVertex = swPoint
        
        Else
        
            Set oConvertInput.swSketchPoint = swPoint
            Set oConvertInput.swVertex = Nothing
        
        End If
        
        Set oConvertInput.PointComp = swSelect.GetSelectedObjectsComponent4(1, -1)

            
    Else
                
        Me.pointTextBox.Value = "Point Not Selected"
        Set oConvertInput.swVertex = Nothing
        Set oConvertInput.swSketchPoint = Nothing
        Me.pointTextBox.BackColor = vbRed
            
    End If
        
End Function

Private Sub CloseButton_Click()
    
    Unload convertSketchForm
    
End Sub

Private Sub OEMDetailsButton_Click()

    If oConvertInput.swSketchPoint Is Nothing And oConvertInput.swVertex Is Nothing Then
    
        MsgBox "Point Not Selected! Please select a Point to Extract OEM Details", vbExclamation
    
    ElseIf IsEmpty(oConvertInput.vCompArr) Then
        
        MsgBox "Please select few OEMs to Extract OEM Details", vbExclamation
    
    Else
    
        If UBound(oConvertInput.vCompArr) < 0 Then
        
            MsgBox "Please select few OEMs to Extract OEM Details", vbExclamation
            
        Else
            oConvertInput.ProjectNo = Me.ProjectNoBox
            Me.Hide
            Call extractOEMDetails(oConvertInput)
            Unload Me
            
        End If
        
    End If
End Sub

Private Sub removeCompButton_Click()

    Dim i As Integer
    
    Dim swModel As SldWorks.ModelDoc2
    Set swModel = swApp.ActiveDoc
    
    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swModel.SelectionManager
    
    If swSelect.GetSelectedObjectCount2(-1) > 0 Then
    
        For i = 1 To swSelect.GetSelectedObjectCount2(-1)
        
            Dim swComp As SldWorks.Component2
            Set swComp = swSelect.GetSelectedObjectsComponent4(i, -1)
            
            If (compDict.Exists(swComp.Name2)) Then
                
                RemoveItemFromList swComp.Name2
            
            End If
        
        Next i
        
         Call AddCompToObject
    
    Else
    
        With convertSketchForm.CompListButton
        
            Dim isListItemSelected As Boolean
            isListItemSelected = False
            
            For i = .ListCount - 1 To 0 Step -1
            
                If .Selected(i) = True Then
                    
                    isListItemSelected = True
                    compDict.Remove .List(i, 0)
                    .RemoveItem (i)
                    
                End If
                
            Next i
            
        End With
            
        If isListItemSelected = True Then
            
            Call AddCompToObject
            
        Else
        
            MsgBox "No items selected to remove from the list", vbInformation
            
        End If
        
    End If
    

End Sub

Private Sub clearListButton_Click()
        
    Dim i As Integer
    With convertSketchForm.CompListButton

        For i = .ListCount - 1 To 0 Step -1
                         
            compDict.Remove .List(i, 0)
            .RemoveItem (i)
                    
        Next i
        
    End With
    
    Call AddCompToObject

End Sub


Private Sub UserForm_Initialize()

    Set oConvertInput = New ISketchConvertInput
    Set compDict = New Scripting.Dictionary

End Sub

Private Function RemoveItemFromList(ItemName As String)
    
    Dim i As Integer
    With convertSketchForm.CompListButton

        For i = .ListCount - 1 To 0 Step -1
                
            If .List(i, 0) = ItemName Then
                        
                compDict.Remove ItemName
                .RemoveItem (i)
                        
            End If
                    
        Next i
        
    End With
    
End Function


