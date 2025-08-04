VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ICutlist"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False

Private CutlistFeat As SldWorks.Feature
Private CutlistFolder As SldWorks.BodyFolder
Private CutlistPropMgr As SldWorks.CustomPropertyManager
Private cPart As SldWorks.ModelDoc2

Property Get GetCutlistFolder() As SldWorks.BodyFolder

    Set GetCutlistFolder = CutlistFolder
    
End Property


Property Get Description() As String

    Description = GetProperty("DESCRIPTION")
    
End Property

Property Get Length() As Double
    
    Length = GetValOfProperty("LENGTH")
    
End Property

Property Get Width() As Double
    
    If CutlistFolder.GetCutListType = swCutListType_e.swWeldmentCutlist Then
        
        Width = ""
    
    Else

        Width = GetValOfProperty("WIDTH")

    End If

End Property

Property Get GetCutlistFeature() As SldWorks.Feature
    
    Set GetCutlistFeature = CutlistFeat
    
End Property

Public Sub Init(Feat As SldWorks.Feature, Part As SldWorks.PartDoc)

    Set CutlistFeat = Feat
    Set CutlistFolder = CutlistFeat.GetSpecificFeature2

    Set CutlistPropMgr = CutlistFeat.CustomPropertyManager
    
    Set cPart = Part

End Sub

Public Function GetProperty(PropName As String) As String
    
    Dim Val As String
    CutlistPropMgr.Get5 PropName, False, Val, GetProperty, True

End Function

Public Function SetProperty(PropName As String, PropValue As String)
    
    CutlistPropMgr.Add3 PropName, swCustomInfoType_e.swCustomInfoText, PropValue, swCustomPropertyAddOption_e.swCustomPropertyDeleteAndAdd
    
End Function


Public Function IsContainThisBody(swBody As SldWorks.Body2) As Boolean

    IsContainThisBody = False
    vBodies = GetBodies
    
    Dim i As Integer
    For i = LBound(vBodies) To UBound(vBodies)
    
        Dim swCutlistBody As SldWorks.Body2
        Set swCutlistBody = vBodies(i)
        
        If swCutlistBody.Name = swBody.Name Then
        
            IsContainThisBody = True
            Exit Function
            
        End If
    
    Next i
    
End Function

Public Function GetBodies()

    GetBodies = CutlistFolder.GetBodies
    
End Function

Private Function GetValOfProperty(PropName As String) As Double

    Dim TempVal As String
    TempVal = GetProperty(PropName)
    
    If IsNumeric(TempVal) Then
    
        GetValOfProperty = CDbl(TempVal)
        
    Else
    
        Dim swUserUnit As SldWorks.UserUnit
        Set swUserUnit = cPart.GetUserUnit(swUserUnitsType_e.swLengthUnit)
        
        swUserUnit.ConvertToSystemValue TempVal, GetValOfProperty
    
    End If

End Function




