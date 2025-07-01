Attribute VB_Name = "AddTables"
Const SheetBorderTop As Double = 0.27030866
Const SheetBorderLeft As Double = 0.01590679
Const SheetBorderRight As Double = 0.41595679

Sub AddViewAndWeldTable(swComp As SldWorks.Component2, swDrawing As SldWorks.DrawingDoc, ViewMaxLoc As Double)

    Dim swDummyInsView As SldWorks.View
    Set swDummyInsView = swDrawing.CreateDrawViewFromModelView3(swComp.GetModelDoc2().GetPathName(), "*Front", 0.206, 0.296, 0)
        
    If Not swDummyInsView Is Nothing Then
        
        Dim swWeldTableAnn As SldWorks.WeldmentCutListAnnotation
        Set swWeldTableAnn = swDummyInsView.InsertWeldmentTable(False, 0.01590679, SheetBorderTop, _
                    swBOMConfigurationAnchorType_e.swBOMConfigurationAnchor_TopLeft, "", "C:\FBD\COMMON\FBD Templates\CUTLIST TABLE.sldwldtbt")
                    
        If Not swWeldTableAnn Is Nothing Then
            
            Dim swTableAnn As SldWorks.TableAnnotation
            Set swTableAnn = swWeldTableAnn
                
            Dim swAnn As SldWorks.Annotation
            Set swAnn = swTableAnn.GetAnnotation
            
'            Call swTableAnn.InsertColumn2(swTableItemInsertPosition_Last, 0, "WIDTH", swInsertColumn_DefaultWidth)
'            Call swTableAnn.SetColumnType(swTableAnn.ColumnCount - 1, swTableColumnTypes_e.swWeldTableColumnType_CustomProperty)
'            Call swTableAnn.SetColumnTitle(swTableAnn.ColumnCount - 1, "WIDTH")
                
            swAnn.Select3 False, Nothing
            
            'swTableAnn.MoveColumn 0, swTableItemInsertPosition_e.swTableItemInsertPosition_After, 1
                
            swWeldTableAnn.Sort 1, True
            'swTableAnn.MoveColumn 1, swTableItemInsertPosition_e.swTableItemInsertPosition_Before, 0

            Call SplitTableIfNeeded(swTableAnn, ViewMaxLoc)

        End If
        
    End If

End Sub
Private Sub SplitTableIfNeeded(swTableAnn As SldWorks.TableAnnotation, ViewMaxLoc As Double)
    

    Dim TableWidth As Double
    TableWidth = setandGetColumnWidth(swTableAnn)

    Dim rowHeight As Double
    rowHeight = swTableAnn.GetRowHeight(0)
    Debug.Print swTableAnn.Text(1, 2)
    
    Dim ViewTopGap As Double
    ViewTopGap = SheetBorderTop - ViewMaxLoc - 0.01
    
    
    Dim i As Integer
    Dim NoOfRows As Integer
    NoOfRows = Int(ViewTopGap / rowHeight)
        
    Dim MaxNoOfSplits As Integer
    MaxNoOfSplits = Int((0.41595679 - 0.01590679) / TableWidth)
        
    If Int(swTableAnn.RowCount / NoOfRows) < MaxNoOfSplits Then
            
        MaxNoOfSplits = Int(swTableAnn.RowCount / NoOfRows)
            
    Else
            
        NoOfRows = Int(swTableAnn.RowCount / (MaxNoOfSplits + 1)) + 1
            
    End If
        
    If Abs(swTableAnn.RowCount - NoOfRows) > 2 Then
        
        For i = 1 To MaxNoOfSplits
    
            Set swTableAnn = swTableAnn.Split(swTableSplitLocations_e.swTableSplit_AfterRow, i * (NoOfRows - 1))
                    
            If Not swTableAnn Is Nothing Then
                    
                Dim swAnn As SldWorks.Annotation
                Set swAnn = swTableAnn.GetAnnotation()
                        
                swAnn.SetPosition2 0.01590679 + i * (TableWidth + 0.005), SheetBorderTop, 0
                        
            End If
 
        Next i
            
    End If

End Sub

Private Function setandGetColumnWidth(swTable As SldWorks.TableAnnotation) As Double
    
    setandGetColumnWidth = 0
    swTable.SetRowHeight swTableCellRangeIdentifier_e.swTableCellRange_All, 0.004, _
        swTableRowColSizeChangeBehavior_e.swTableRowColChange_TableSizeCanChange
    Const SingleTextWidth = 0.0028
    
    Dim i As Integer
    For i = 0 To swTable.ColumnCount - 1
        
        swTable.setColumnWidth i, SingleTextWidth * Len(swTable.Text(0, i)), _
                swTableRowColSizeChangeBehavior_e.swTableRowColChange_TableSizeCanChange
                
        setandGetColumnWidth = setandGetColumnWidth + swTable.GetColumnWidth(i)
        
    Next i

End Function
Private Function GetColIdx(ColName As String, swTable As SldWorks.TableAnnotation)

    Dim i As Integer
    For i = 0 To swTable.ColumnCount - 1
        
        If swTable.Text(0, i) = ColName Then
        
            GetColIdx = i
            Exit For
            
        End If
    
    Next i
    
End Function
Private Function GetTableWidth(swTable As SldWorks.TableAnnotation) As Double

    GetTableWidth = 0
    
    Dim i As Integer
    For i = 0 To swTable.ColumnCount - 1
        
        Debug.Print swTable.GetColumnWidth(i)
        GetTableWidth = GetTableWidth + swTable.GetColumnWidth(i)
            
    Next i
    
End Function

