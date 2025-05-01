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

Property Get IsDoor() As Boolean

    IsDoor = False
    
End Property


Property Get UpperLeftLTab() As ILTabs
    
    If Not IsEmpty(LTabsList.Items) And UBound(LTabsList.Items) >= 1 Then
    
        Set UpperLeftLTab = LTabsList.Items(1)
        
    End If
    
End Property

Property Get UpperRightLTab() As ILTabs
    
    If Not IsEmpty(LTabsList.Items) And UBound(LTabsList.Items) = 3 Then
    
        Set UpperRightLTab = LTabsList.Items(3)
        
    End If
    
End Property

Property Get LowerRightLTab() As ILTabs
    
    If Not IsEmpty(LTabsList.Items) And UBound(LTabsList.Items) >= 2 Then
    
        Set LowerRightLTab = LTabsList.Items(2)
        
    End If
    
End Property

Property Get LowerLeftLTab() As ILTabs
    
    If Not IsEmpty(LTabsList.Items) Then
    
        Set LowerLeftLTab = LTabsList.Items(LBound(LTabsList.Items))
        
    End If
    
End Property

Property Get LowerLeftXPoint() As Double
    
    If Not IsEmpty(LTabsList.Items) Then
    
        LowerLeftXPoint = LowerLeftLTab.xPoint
        
    End If
    
End Property
Property Get LowerLeftYPoint() As Double
    
    If Not IsEmpty(LTabsList.Items) Then
    
        LowerLeftYPoint = LowerLeftLTab.yPoint
        
    End If
    
End Property

Property Get UpperRightYPoint() As Double
    
    If Not IsEmpty(LTabsList.Items) Then
    
        UpperRightYPoint = UpperRightLTab.yPoint
        
    End If
    
End Property
Property Get UpperRightXPoint() As Double
    
    If Not IsEmpty(LTabsList.Items) Then
    
        UpperRightXPoint = UpperRightLTab.xPoint
        
    End If
    
End Property


Sub Initialize(oLTab As ILTabs, oLNextTab As ILTabs)

    Set LTabsList = New IArrListObject
    LTabsList.AddtoList oLTab
    LTabsList.AddtoList oLNextTab

End Sub

Sub AddToTabsList(oLTab As ILTabs, oLNextTab As ILTabs)

    LTabsList.AddtoList oLTab
    LTabsList.AddtoList oLNextTab
    
End Sub


