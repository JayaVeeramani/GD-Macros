VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IOEM"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cComp As SldWorks.Component2
Private cModel As SldWorks.ModelDoc2

Private cConfig As IConfiguration
Private cCustPropMgr As SldWorks.CustomPropertyManager
Private cDesc As String

Private posVector As SldWorks.MathVector
Private cFileName As String
Private cPartNumber As String
Private cWeight As Double

Private cXSize As Double
Private cYSize As Double
Private cZSize As Double

Property Get FileName() As String

    FileName = cFileName
    
End Property

Property Get PartNumber() As String

    PartNumber = cPartNumber
    
End Property

Property Get Description() As String

    Description = cDesc
    
End Property

Property Get Weight() As Double

    Weight = cWeight
    
End Property


Property Get xPos() As Double

    xPos = posVector.ArrayData(0)

End Property

Property Get yPos() As Double

    yPos = posVector.ArrayData(1)

End Property

Property Get zPos() As Double

    zPos = -1 * posVector.ArrayData(2)

End Property

Property Get xSize() As Double

    xSize = cXSize
    
End Property

Property Get ySize() As Double

    ySize = cYSize
    
End Property

Property Get zSize() As Double
    
    zSize = cZSize

End Property


Sub Initialize(swComp As IComponent2, swOriginVector As SldWorks.MathVector)

    Set cComp = swComp
    Set cModel = ResolveAndGetModelDoc(swComp)
    
    Set cConfig = cModel.ConfigurationManager.ActiveConfiguration
    Set cCustPropMgr = cConfig.CustomPropertyManager
    
    Dim wasResolved As Boolean
    Dim Val As String
    cCustPropMgr.Get5 "Description", False, Val, cDesc, wasResolved
    
    Dim vSplitArr As Variant
    vSplitArr = Split(swComp.GetPathName, "\")
    
    Dim TempFileName As String
    TempFileName = vSplitArr(UBound(vSplitArr))
    
    cFileName = Left(TempFileName, InStr(TempFileName, ".") - 1)
    cPartNumber = BOMPartNumber
    cWeight = GetWeight
    
    Call GetComponentPositionVector(swOriginVector)

End Sub

Sub GetComponentPositionVector(swOriginVector As SldWorks.MathVector)

    Dim vBox As Variant
    vBox = cComp.GetBox(False, False)
        
    Dim dPoint(2) As Double
    Dim vPoint As Variant
    
    dPoint(0) = vBox(0)
    dPoint(1) = vBox(1)
    dPoint(2) = vBox(5)
    
    cXSize = (vBox(3) - vBox(0)) / 0.0254
    cYSize = (vBox(4) - vBox(1)) / 0.0254
    cZSize = (vBox(5) - vBox(2)) / 0.0254
       
    vPoint = dPoint
       
    Dim swCompVector As SldWorks.MathVector
    Set swCompVector = swMathUtility.CreateVector(vPoint)
       
    Dim swPositionVector As SldWorks.MathVector
    Set swPositionVector = swCompVector.Subtract(swOriginVector)

    Set posVector = swPositionVector.Scale(1 / 0.0254)
    
End Sub


Private Function ResolveAndGetModelDoc(swComp As SldWorks.Component2) As SldWorks.ModelDoc2

    If swComp.GetSuppression = swComponentSuppressionState_e.swComponentLightweight Then

        Dim bRet As Integer
        bRet = swComp.SetSuppression2(swComponentSuppressionState_e.swComponentResolved)

    End If
    
    Set ResolveAndGetModelDoc = swComp.GetModelDoc2()
    
End Function

Function BOMPartNumber()

    Select Case cConfig.BOMPartNoSource
        Case SwConst.swBOMPartNumberSource_e.swBOMPartNumber_ConfigurationName
            BOMPartNumber = cConfig.Name
        Case SwConst.swBOMPartNumberSource_e.swBOMPartNumber_DocumentName
            BOMPartNumber = cFileName
        Case SwConst.swBOMPartNumberSource_e.swBOMPartNumber_UserSpecified
            BOMPartNumber = cConfig.AlternateName
    End Select
    
End Function

Function GetWeight() As Double

    Dim swMassProp As IMassProperty
    Set swMassProp = cModel.Extension.CreateMassProperty()
    
    GetWeight = Round((swMassProp.Mass * 2.2046226218), 2)
    
End Function
