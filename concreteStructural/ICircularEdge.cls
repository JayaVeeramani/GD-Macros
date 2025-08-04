VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ICircularEdge"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cEdge As SldWorks.Edge
Private cComp As SldWorks.Component2

Private vCircleCenter As Variant
Private vViewCenter As Variant





Property Get GetEdge() As SldWorks.Edge

    Set GetEdge = cEdge
    
End Property


Property Get xMin() As Double

    xMin = vCircleCenter(0)
    
End Property

Property Get yMin() As Double

    yMin = vCircleCenter(1)
    
End Property

Property Get xViewMin() As Double

    xViewMin = vViewCenter(0)
    
End Property


Property Get yViewMin() As Double

    yViewMin = vViewCenter(1)
    
End Property


Sub Initialize(swEdge As SldWorks.Edge, vCircleParams As Variant, swView As SldWorks.View, swComp As SldWorks.Component2)

    Set cEdge = swEdge
    Set cComp = swComp
    
    Dim dCenterPt(2) As Double
    dCenterPt(0) = vCircleParams(0)
    dCenterPt(1) = vCircleParams(1)
    dCenterPt(2) = vCircleParams(2)
    
    vCircleCenter = dCenterPt
    vCircleCenter = GetComponentPointInSheetSpace(swComp, vCircleCenter, swView)
    

    vViewCenter = GetSheetPointInViewSpace(swView, vCircleCenter)
    

End Sub

