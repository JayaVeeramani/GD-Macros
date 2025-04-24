VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} InsulationForm 
   Caption         =   "Select Insulation"
   ClientHeight    =   1716
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   3408
   OleObjectBlob   =   "InsulationForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "InsulationForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False






Private Sub OkButton_Click()

    IsInsulationFormClicked = True
    Unload Me
    
End Sub



Private Sub UserForm_QueryClose(Cancel As Integer, CloseMode As Integer)

    IsInsulationFormClicked = True

End Sub
