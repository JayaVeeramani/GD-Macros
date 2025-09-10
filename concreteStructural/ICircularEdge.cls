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

Private cRadius As Double


Property Get ViewRadius() As Double

    ViewRadius = cRadius
    
End Property


Property Get GetEdge() As SldWorks.Edge

    Set GetEdge = cEdge
    
End Property


Property Get xCenter() As Double

    xCenter = vCircleCenter(0)
    
End Property

Property Get yCenter() As Double

    yCenter = vCircleCenter(1)
    
End Property

Property Get xViewCenter() As Double

    xViewCenter = vViewCenter(0)
    
End Property

Property Get yViewCenter() As Double

    yViewCenter = vViewCenter(1)
    
End Property


Sub Initialize(swEdge As SldWorks.Edge, swView As SldWorks.View, swComp As SldWorks.Component2)

    Set cEdge = swEdge
    Set cComp = swComp
    
    Dim swCurve As SldWorks.Curve
    Set swCurve = cEdge.GetCurve

    Dim vCircleParams As Variant
    vCircleParams = swCurve.CircleParams
    
    cRadius = vCircleParams(6) * swView.ScaleDecimal
    
    Dim dCenterPt(2) As Double
    dCenterPt(0) = vCircleParams(0)
    dCenterPt(1) = vCircleParams(1)
    dCenterPt(2) = vCircleParams(2)
    
    vCircleCenter = dCenterPt
    vViewCenter = GetComponentPointInSheetSpace(swComp, vCircleCenter, swView)


End Sub

