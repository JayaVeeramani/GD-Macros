VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} SubAssyForm 
   Caption         =   "Select Face or Components"
   ClientHeight    =   1716
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   3408
   OleObjectBlob   =   "SubAssyForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "SubAssyForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False

Private Sub OkButton_Click()

    IsSubAssyFormClicked = True
    Unload Me
    
End Sub


