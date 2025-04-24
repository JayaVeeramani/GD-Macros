VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} ViewNameForm 
   Caption         =   "Select View"
   ClientHeight    =   1644
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   3408
   OleObjectBlob   =   "ViewNameForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "ViewNameForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False




Option Explicit

Private Sub CloseButton_Click()
    
    Unload Me
    
End Sub

Private Sub UserForm_Initialize()

    With Me.ViewNameBox
    
        Dim vModelViewNames As Variant
        vModelViewNames = swTopLevelModel.GetModelViewNames

        Dim i As Integer
        For i = LBound(vModelViewNames) To UBound(vModelViewNames)

            .AddItem vModelViewNames(i)

        Next i

        
    End With
    
   'Me.WallNameBox.Text = "WALL-X"

End Sub



Private Sub ViewButton_Click()

    Me.Hide
    
End Sub

Private Sub ViewNameBox_Change()

End Sub
