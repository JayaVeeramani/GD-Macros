VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IContourSketch"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False

Private oContour As SldWorks.SketchContour

'Private cxMin As Double
'Private cxMax As Double
'
'Private czMin As Double
'Private czMax As Double

Private bLeftPoint As SldWorks.sketchPoint
Private bRightPoint As SldWorks.sketchPoint
Private tLeftPoint As SldWorks.sketchPoint
Private tRightPoint As SldWorks.sketchPoint

Private topSegment As SldWorks.SketchSegment
Private bottomSegment As SldWorks.SketchSegment
Private leftSegment As SldWorks.SketchSegment
Private rightSegment As SldWorks.SketchSegment

Private cisRectangular As Boolean

Public BalloonLegend As String
Public BalloonId As String

Property Get Length() As Double

    If cisRectangular Then
    
        Length = topSegment.GetLength
        
    End If
    
End Property

Property Get Width() As Double

    If cisRectangular Then
    
        Width = leftSegment.GetLength
        
    End If

End Property

Property Get isRectangular() As Boolean

    isRectangular = cisRectangular
    
End Property

Property Get bottomLeftPoint() As SldWorks.sketchPoint
    
    Set bottomLeftPoint = bLeftPoint

End Property

Property Get BottomRightPoint() As SldWorks.sketchPoint
    
    Set BottomRightPoint = bRightPoint

End Property

Property Get TopLeftPoint() As SldWorks.sketchPoint
    
    Set TopLeftPoint = tLeftPoint

End Property

Property Get topRightPoint() As SldWorks.sketchPoint
    
    Set topRightPoint = tRightPoint

End Property


Property Get TopSketchLine() As SldWorks.SketchSegment
    
    Set TopSketchLine = topSegment

End Property

Property Get BottomSketchLine() As SldWorks.SketchSegment

    Set BottomSketchLine = bottomSegment

End Property

Property Get LeftSketchLine() As SldWorks.SketchSegment

    Set LeftSketchLine = leftSegment
    
End Property

Property Get RightSketchLine() As SldWorks.SketchSegment

    Set RightSketchLine = rightSegment
    
End Property

Public Sub Initialize(swContour As SldWorks.SketchContour)
    
    Set oContour = swContour
    cisRectangular = GetRectangularLimits

End Sub

Private Function GetRectangularLimits() As Boolean

    GetRectangularLimits = True
    
    Dim vSketchSegments As Variant
    vSketchSegments = oContour.GetSketchSegments
    
    Dim xCount As Integer
    Dim yCount As Integer
    xCount = 0
    yCount = 0
    
    Dim tempVerticalLine As SldWorks.SketchSegment
    Dim tempYMinSketchPoint As SldWorks.sketchPoint
    Dim tempYMaxSketchPoint As SldWorks.sketchPoint
    Dim tempXVal As Double

    Dim tempHorizontalLine As SldWorks.SketchSegment
    Dim tempYVal As Double

    
    Dim i As Integer
    For i = LBound(vSketchSegments) To UBound(vSketchSegments)
            
        Dim swSketchSegment As SldWorks.SketchSegment
        Set swSketchSegment = vSketchSegments(i)
            
        Dim swSketchLine As SldWorks.SketchLine
        Set swSketchLine = swSketchSegment

        Dim swStartPoint As SldWorks.sketchPoint
        Set swStartPoint = swSketchLine.GetStartPoint2
        
        Dim swEndPoint As SldWorks.sketchPoint
        Set swEndPoint = swSketchLine.GetEndPoint2
        
        If Abs(swStartPoint.X - swEndPoint.X) <= 0.000001 Then
        
            xCount = xCount + 1
            If xCount = 1 Then
            
                Set tempVerticalLine = swSketchSegment
                tempXVal = swStartPoint.X
                Call GetMinMaxYSketchPoint(swStartPoint, swEndPoint, tempYMinSketchPoint, tempYMaxSketchPoint)

            ElseIf xCount = 2 Then
            
                If tempXVal < swStartPoint.X Then
                    
                    Set leftSegment = tempVerticalLine
                    Set rightSegment = swSketchSegment
                    Set bLeftPoint = tempYMinSketchPoint
                    Set tLeftPoint = tempYMaxSketchPoint
                    Call GetMinMaxYSketchPoint(swStartPoint, swEndPoint, bRightPoint, tRightPoint)
                
                Else
                
                    Set leftSegment = swSketchSegment
                    Set rightSegment = tempVerticalLine
                    Set bRightPoint = tempYMinSketchPoint
                    Set tRightPoint = tempYMaxSketchPoint
                    Call GetMinMaxYSketchPoint(swStartPoint, swEndPoint, bLeftPoint, tLeftPoint)
                    
                End If
            
            Else
            
                GetRectangularLimits = False
                Exit Function
                
            End If
        
        ElseIf Abs(swStartPoint.Y - swEndPoint.Y) <= 0.000001 Then
            
            yCount = yCount + 1
            If yCount = 1 Then
                
                Set tempHorizontalLine = swSketchSegment
                tempYVal = swStartPoint.Y
            
            ElseIf yCount = 2 Then
                
                If tempYVal < swStartPoint.Y Then
                
                    Set bottomSegment = tempHorizontalLine
                    Set topSegment = swSketchSegment
                
                Else
                
                    Set bottomSegment = swSketchSegment
                    Set topSegment = tempHorizontalLine
                
                End If
            
            Else
            
                GetRectangularLimits = False
                Exit Function
                
            End If
        
        Else
        
            GetRectangularLimits = False
            Exit Function
            
        End If

    Next i
    
    If (bLeftPoint Is Nothing Or bRightPoint Is Nothing _
            Or tLeftPoint Is Nothing Or tRightPoint Is Nothing) Then
            
        GetRectangularLimits = False
        Exit Function
        
    End If

'        Dim swSketch As SldWorks.Sketch
'        Set swSketch = bLeftPoint.GetSketch
'
'        Dim swSketchToModelTransform As SldWorks.MathTransform
'        Set swSketchToModelTransform = swSketch.ModelToSketchTransform.Inverse
'
'        Dim bLeftMathpoint As SldWorks.MathPoint
'        Dim tRightMathpoint As SldWorks.MathPoint
'
'        Set bLeftMathpoint = GetSketchToModelMathPoint(swSketchToModelTransform, bLeftPoint)
'        Set tRightMathpoint = GetSketchToModelMathPoint(swSketchToModelTransform, tRightPoint)
'
'        cxMin = bLeftMathpoint.ArrayData(0)
'        czMin = bLeftMathpoint.ArrayData(2)
'
'        cxMax = tRightMathpoint.ArrayData(0)
'        czMax = tRightMathpoint.ArrayData(2)
'
'    Else
'
'            GetRectangularLimits = False
'            Exit Function


End Function


Sub GetMinMaxYSketchPoint(swPoint1 As SldWorks.sketchPoint, swPoint2 As SldWorks.sketchPoint, _
            ByRef tempYMinSketchPoint As SldWorks.sketchPoint, ByRef tempYMaxSketchPoint As SldWorks.sketchPoint)
    
    If swPoint1.Y < swPoint2.Y Then
        
        Set tempYMinSketchPoint = swPoint1
        Set tempYMaxSketchPoint = swPoint2
    
    Else
    
        Set tempYMinSketchPoint = swPoint2
        Set tempYMaxSketchPoint = swPoint1
    
    End If
    
End Sub

'Function GetSketchToModelMathPoint(xForm As SldWorks.MathTransform, point As SldWorks.SketchPoint) As SldWorks.MathPoint
'
'    Dim PointVal(2) As Double
'    PointVal(0) = point.X: PointVal(1) = point.Y: PointVal(2) = point.Z
'
'    Dim swMathUtility As SldWorks.MathUtility
'    Set swMathUtility = swApp.GetMathUtility
'
'    Set GetSketchToModelMathPoint = swMathUtility.CreatePoint(PointVal)
'    Set GetSketchToModelMathPoint = GetSketchToModelMathPoint.MultiplyTransform(xForm)
'
'End Function

