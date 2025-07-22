Attribute VB_Name = "AddHatch"

Sub AddHatchAndCallOutForFoams(oFoamBody As IWeldBody, swDrawing As SldWorks.DrawingDoc, _
    swView As SldWorks.View)

    Dim swFace As SldWorks.Face2
    Set swFace = oFoamBody.NormalFace
            
    Dim xPos As Double
    xPos = (oFoamBody.xMax + oFoamBody.xMin) / 2
            
    Dim yPos As Double
    yPos = (oFoamBody.yMax + oFoamBody.yMin) / 2
            
    Call SelectAndAddItemNoAnnotation(swFace, swDrawing, swView, xPos, yPos, xPos - 0.002, yPos + 0.00125)

    swView.SelectEntity swFace, False
    swDrawing.InsertHatchedFace

End Sub

Sub UpdateHatchProperties(swView As SldWorks.View)
    
    Dim swSketch As SldWorks.Sketch
    Set swSketch = swView.GetSketch
    
    Dim vSketchHatches As Variant
    vSketchHatches = swSketch.GetSketchHatches
            
    If Not IsEmpty(vSketchHatches) Then
            
        Dim i As Integer
        For i = LBound(vSketchHatches) To UBound(vSketchHatches)
                
            Dim swSketchHatch As SldWorks.SketchHatch
            Set swSketchHatch = vSketchHatches(i)
                    
            swSketchHatch.Pattern = "Honeycomb"
            swSketchHatch.Scale2 = swView.ScaleDecimal * 2
            
            swSketchHatch.Layer = HatchLayName
                
        Next i
                
    End If

End Sub

Function GetLargestFace(vFaces As Variant) As SldWorks.Face2

    Dim i As Integer
    Dim Area As Double
    Area = 0
    For i = LBound(vFaces) To UBound(vFaces)
    
        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(i)
        
        If swFace.GetArea > Area Then
        
            Set GetLargestFace = swFace
            Area = swFace.GetArea
            
        End If

    Next i
   
End Function

Function GetNormalFaces(vFaces As Variant, CompTransform As IMathTransform, _
    swViewNormalVector As SldWorks.MathVector) As Variant
    
    Dim FaceCount As Integer
    FaceCount = 0
    
    Dim NormalFaces() As SldWorks.Face2

    Dim i As Integer
    For i = LBound(vFaces) To UBound(vFaces)

        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(i)

        Dim swSurface As SldWorks.Surface
        Set swSurface = swFace.GetSurface
        
        Set swViewNormalVector = swViewNormalVector.Normalise
        
        Dim swFaceNormalVector As SldWorks.MathVector
        Set swFaceNormalVector = swMathUtility.CreateVector(swFace.Normal)
        
        Set swFaceNormalVector = swFaceNormalVector.MultiplyTransform(CompTransform)
        Set swFaceNormalVector = swFaceNormalVector.Normalise
        
        Dim Angle As Double
        Dim DotProduct As Double
        DotProduct = swFaceNormalVector.Dot(swViewNormalVector)
        
        If DotProduct >= 1 Then
        
            Angle = Arccos(Int(DotProduct)) * 180# / 3.14159265359
            
        ElseIf DotProduct < -1 Then
        
            Angle = Arccos(Int(DotProduct) + 1) * 180# / 3.14159265359
        
        Else
        
            Angle = Arccos(DotProduct) * 180# / 3.14159265359
            
        End If
        
        If Not swSurface Is Nothing Then
 
        If swSurface.IsPlane And Angle <= 0.01 Then
            
            Dim swEnt As SldWorks.Entity
            Set swEnt = swFace
            Set swEnt = swEnt.GetSafeEntity
            
            ReDim Preserve NormalFaces(FaceCount)
            Set NormalFaces(FaceCount) = swEnt
            FaceCount = FaceCount + 1
            
        End If
        
        End If

    Next i
    
    GetNormalFaces = NormalFaces

End Function
