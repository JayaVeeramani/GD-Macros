VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} ViewNameForm 
   Caption         =   "Select View"
   ClientHeight    =   2076
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
    
        .AddItem "*Front"
        .AddItem "*Back"
        .AddItem "*Left"
        .AddItem "*Right"
        .AddItem "*Top"
        .AddItem "*Bottom"
        
    End With
    
   Me.WallNameBox.Text = "WALL-X"

End Sub



Private Sub ViewButton_Click()

    Me.Hide
    
End Sub
