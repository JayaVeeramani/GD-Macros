Attribute VB_Name = "extract_OEMDetails"

Public swApp As SldWorks.SldWorks
Public swMathUtility As SldWorks.MathUtility

Dim swModel As SldWorks.ModelDoc2

Const TextHeight As Double = 0.009525
Const Macro_Name As String = "OEM Details Extracter"

Sub main()

    Set swApp = Application.SldWorks
    Set swModel = swApp.ActiveDoc
    
    Set swMathUtility = swApp.GetMathUtility()
    
    If swModel.GetType = swDocumentTypes_e.swDocASSEMBLY Then 'swDocumentTypes_e.swDocASSEMBLY Then
    
        convertSketchForm.ProjectNoBox.Value = Mid(swModel.GetPathName, InStrRev(swModel.GetPathName, "\") + 1, 6)
        convertSketchForm.Show vbModeless
    
    Else
        
        MsgBox "Please open assembly doc and run the macro"
        
    End If

End Sub

Function extractOEMDetails(oConvertInput As ISketchConvertInput)

    Dim swOriginVector As SldWorks.MathVector
    Set swOriginVector = GetPositionVectorWithRespectToOrigin(oConvertInput)

    Dim vComps As Variant
    vComps = oConvertInput.vCompArr
    
    Dim vOEMComps As Variant
    ReDim vOEMComps(UBound(vComps))
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
        
        Dim swComp As SldWorks.Component2
        Set swComp = vComps(i)

        Dim OEMComp As IOEM
        Set OEMComp = New IOEM
        
        OEMComp.Initialize swComp, swOriginVector
         
        Set vOEMComps(i) = OEMComp
       
    Next i
    
    Call GenerateOEMDetailsBOM(vOEMComps, oConvertInput.ProjectNo)

End Function

Function GetPositionVectorWithRespectToOrigin(oConvertInput As ISketchConvertInput) As SldWorks.MathVector

    Dim vPoint As Variant
    If Not (oConvertInput.swVertex Is Nothing) Then
    
        Dim swVertex As SldWorks.Vertex
        Set swVertex = oConvertInput.swVertex

        vPoint = swVertex.GetPoint
        
    ElseIf Not (oConvertInput.swSketchPoint Is Nothing) Then
    
        Dim swSketchPoint As SldWorks.sketchPoint
        Set swSketchPoint = oConvertInput.swSketchPoint

        
        Dim dPoint(2) As Double
        dPoint(0) = swSketchPoint.X
        dPoint(1) = swSketchPoint.Y
        dPoint(2) = swSketchPoint.Z
        
        vPoint = dPoint
        
    End If
    
    Dim swOwningComp As SldWorks.Component2
    Set swOwningComp = oConvertInput.PointComp
    
    vPoint = GetTransformPoint(vPoint, swOwningComp.Transform2)
    Set GetPositionVectorWithRespectToOrigin = swMathUtility.CreateVector(vPoint)
        
End Function

Function GetTransformPoint(vPoint As Variant, swTransform As SldWorks.MathTransform)
    
    Dim swMathPoint As SldWorks.MathPoint
    Set swMathPoint = swMathUtility.CreatePoint(vPoint)
    
    Set swMathPoint = swMathPoint.MultiplyTransform(swTransform)
    GetTransformPoint = swMathPoint.ArrayData

End Function


Private Sub GenerateOEMDetailsBOM(vConsolidatedList As Variant, ProjectNo As String)

    Dim xlApp As Excel.Application
    Set xlApp = New Excel.Application
    xlApp.Visible = False

    Dim xlWB As Excel.Workbook
    Set xlWB = xlApp.Workbooks.Add

    Dim XlSheet As Excel.Worksheet
    Set XlSheet = xlWB.ActiveSheet

    With XlSheet

        .Cells(1, 1).Value = "Project Number"
        .Cells(1, 2).Value = "File Name"
        .Cells(1, 3).Value = "Part Number"
        .Cells(1, 4).Value = "Description"
        .Cells(1, 5).Value = "X Pos"
        .Cells(1, 6).Value = "Y Pos"
        .Cells(1, 7).Value = "Z Pos"
        .Cells(1, 8).Value = "X Size"
        .Cells(1, 9).Value = "Y Size"
        .Cells(1, 10).Value = "Z Size"
        .Cells(1, 11).Value = "Weight (lbs)"

        Dim i As Integer
        For i = LBound(vConsolidatedList) To UBound(vConsolidatedList)

            Dim OEMComp As IOEM
            Set OEMComp = vConsolidatedList(i)

            .Cells(2 + i, 1).Value = ProjectNo
            .Cells(2 + i, 2).Value = OEMComp.FileName
            
            .Cells(2 + i, 3).Value = OEMComp.PartNumber
            .Cells(2 + i, 4).Value = OEMComp.Description
            
            
            .Cells(2 + i, 5).Value = OEMComp.xPos
            .Cells(2 + i, 5).NumberFormat = "0.00"
            
            .Cells(2 + i, 6).Value = OEMComp.yPos
            .Cells(2 + i, 6).NumberFormat = "0.00"
            
            .Cells(2 + i, 7).Value = OEMComp.zPos
            .Cells(2 + i, 7).NumberFormat = "0.00"
            
            .Cells(2 + i, 8).Value = OEMComp.xSize
            .Cells(2 + i, 8).NumberFormat = "0.00"
            
            .Cells(2 + i, 9).Value = OEMComp.ySize
            .Cells(2 + i, 9).NumberFormat = "0.00"
            
            .Cells(2 + i, 10).Value = OEMComp.zSize
            .Cells(2 + i, 10).NumberFormat = "0.00"
            
            .Cells(2 + i, 11).Value = OEMComp.Weight

        Next i
        
        .Cells.EntireColumn.AutoFit
        .Cells.EntireRow.AutoFit

    End With

    xlApp.Visible = True

End Sub




