VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ILTabAssy"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Dim LTabsList As IArrListObject

Property Get UpperLeftLTab() As ILTabs
    
    If Not IsEmpty(LTabsList.Items) And UBound(LTabsList.Items) >= 1 Then
    
        Set UpperLeftLTab = LTabsList.Items(1)
        
    End If
    
End Property

Property Get LowerLeftLTab() As ILTabs
    
    If Not IsEmpty(LTabsList.Items) Then
    
        Set LowerLeftLTab = LTabsList.Items(LBound(LTabsList.Items))
        
    End If
    
End Property

Sub Initialize(oLTab As ILTabs, oLNextTab As ILTabs)

    Set LTabsList = New IArrListObject
    LTabsList.AddToList oLTab
    LTabsList.AddToList oLNextTab

End Sub

Sub AddToTabsList(oLTab As ILTabs, oLNextTab As ILTabs)

    LTabsList.AddToList oLTab
    LTabsList.AddToList oLNextTab
    
End Sub


