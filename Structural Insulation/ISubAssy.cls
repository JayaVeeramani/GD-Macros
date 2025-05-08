VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ISubAssy"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False

Public StartEdge As SldWorks.Edge
Public EndEdge As SldWorks.Edge

Public StartIdx As Integer
Public EndIdx As Integer

Public StartComp As IComp
Public EndComp As IComp

Public Dimension As SldWorks.DisplayDimension
Public AssyLength As Double



Property Get yMin() As Double

    If Not StartComp Is Nothing And Not EndComp Is Nothing Then
    
        If StartComp.yMin < EndComp.yMin Then
            yMin = StartComp.yMin
        Else
            yMin = EndComp.yMin
        End If
        
    End If
    
End Property

Property Get yMax() As Double

    If Not StartComp Is Nothing And Not EndComp Is Nothing Then
    
        If StartComp.yMax > EndComp.yMax Then
            yMax = StartComp.yMax
        Else
            yMax = EndComp.yMax
        End If
        
    End If
    
End Property

