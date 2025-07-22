Attribute VB_Name = "AddTables"
Function InsertBOMAndOrderComponents(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, ViewMaxLoc As Double, _
            ByRef TableEndPt As Double) As SldWorks.BomTableAnnotation
    
    Dim swConfig As SldWorks.Configuration
    Set swConfig = swTopLevelModel.GetActiveConfiguration
    
    Dim configName As String
    configName = swConfig.Name

    Set InsertBOMAndOrderComponents = swView.InsertBomTable2(False, SheetBorderRight, SheetBorderTop, swBOMConfigurationAnchorType_e.swBOMConfigurationAnchor_TopRight, _
                    swBomType_e.swBomType_PartsOnly, configName, "C:\FBD\COMMON\FBD Templates\SIMPLE BOM.sldbomtbt")
                    
    
    If Not InsertBOMAndOrderComponents Is Nothing Then
    
        Dim swTableAnn As SldWorks.TableAnnotation
        Set swTableAnn = InsertBOMAndOrderComponents
        
        Call OrderBOMTable(swTableAnn)
        Call SplitTableIfNeeded(swTableAnn, ViewMaxLoc, TableEndPt)
    
    End If
    
End Function

Private Sub OrderBOMTable(swTableAnn As SldWorks.TableAnnotation)
    
    Dim i As Integer
    For i = 1 To swTableAnn.rowCount - 1
    
        Dim Desc As String
        Desc = swTableAnn.DisplayedText(i, 2)
        
        Select Case True
         
            Case InStr(Desc, "WIRE") > 0 And InStr(Desc, "MESH") > 0
            
                Call MoveTableRow(swTableAnn, i, 1)
                
            Case InStr(Desc, "#3") > 0 And InStr(Desc, "REBAR") And Not (InStr(Desc, "BEND") > 0)
                
                Call MoveTableRow(swTableAnn, i, 3)
                
            Case InStr(Desc, "#4") > 0 And InStr(Desc, "REBAR") > 0
            
                Call MoveTableRow(swTableAnn, i, 4)
            
            Case InStr(Desc, "#5") > 0 And InStr(Desc, "REBAR") > 0

                Call MoveTableRow(swTableAnn, i, 5)
            
            Case InStr(Desc, "#6") > 0 And InStr(Desc, "REBAR") > 0
                
                Call MoveTableRow(swTableAnn, i, 6)
            
            Case InStr(Desc, "FOAM") > 0
                
                If InStr(Desc, "4" & Chr(34)) > 0 Then
                
                    Call MoveTableRow(swTableAnn, i, 8)
                
                Else
                
                    Call MoveTableRow(swTableAnn, i, 7)
                    
                End If

        End Select

    Next i
    
End Sub

Private Sub MoveTableRow(swTableAnn As SldWorks.TableAnnotation, ByRef CurRow As Integer, DestRow As Integer)

    If swTableAnn.rowCount - 1 >= DestRow Then
    
        If Not (DestRow = CurRow) Then
        
            Dim Bool As Boolean
            
            
            If DestRow > CurRow Then
            
                Bool = swTableAnn.MoveRow(CurRow, swTableItemInsertPosition_e.swTableItemInsertPosition_After, DestRow)
                CurRow = CurRow - 1
                
            ElseIf DestRow < CurRow Then
            
                Bool = swTableAnn.MoveRow(CurRow, swTableItemInsertPosition_e.swTableItemInsertPosition_Before, DestRow)
                Bool = swTableAnn.MoveRow(DestRow + 1, swTableItemInsertPosition_e.swTableItemInsertPosition_After, CurRow)
    
                
            End If
            
        End If

    End If
    

End Sub

Private Sub SplitTableIfNeeded(swTableAnn As SldWorks.TableAnnotation, ViewMaxLoc As Double, ByRef TableEndPt As Double)
    
    swTableAnn.setColumnWidth 2, 0.0875, swTableRowColSizeChangeBehavior_e.swTableRowColChange_TableSizeCanChange
    
    Dim TableWidth As Double
    TableWidth = GetColumnWidth(swTableAnn)

    Dim rowHeight As Double
    rowHeight = swTableAnn.GetRowHeight(0)
    Debug.Print swTableAnn.Text(1, 2)
    
'    Dim ViewTopGap As Double
'    ViewTopGap = SheetBorderTop - ViewMaxLoc + 0.02
    
    
    Dim i As Integer
    Dim NoOfRows As Integer
    NoOfRows = Int(ViewTopGap / rowHeight)
        
    Dim MaxNoOfSplits As Integer
    MaxNoOfSplits = Int((SheetBorderRight - SheetBorderLeft) / TableWidth)
    
    Dim GapBwTables As Double
    
    If MaxNoOfSplits > 1 Then
    
        GapBwTables = ((SheetBorderRight - SheetBorderLeft) - (MaxNoOfSplits * TableWidth)) / (MaxNoOfSplits - 1)
        
        If GapBwTables > 0.005 Then
            
            GapBwTables = 0.005
            
        End If
    Else
        
        GapBwTables = 0.005
        
    End If
        

    Dim rowToSplit As Integer
        
    For i = 1 To MaxNoOfSplits
    
        TableEndPt = SheetBorderRight - TableWidth
        rowToSplit = GetRowToSplitAfter(swTableAnn, ViewMaxLoc, rowToSplit)
        
        If rowToSplit = 0 Then
            
            Exit For
            
        Else
            
            Set swTableAnn = swTableAnn.Split(swTableSplitLocations_e.swTableSplit_AfterRow, rowToSplit)
                        
            If Not swTableAnn Is Nothing Then
                        
                Dim swAnn As SldWorks.Annotation
                Set swAnn = swTableAnn.GetAnnotation()
                            
                swAnn.SetPosition2 0.41595679 - i * (TableWidth + GapBwTables), SheetBorderTop, 0
                TableEndPt = 0.41595679 - i * (TableWidth + GapBwTables) - TableWidth
                            
            End If
            
        End If
 
    Next i


End Sub

Private Function GetRowToSplitAfter(swTableAnn As SldWorks.TableAnnotation, ViewMaxLoc As Double, rowStart As Integer)

    Dim TableLoc As Double
    TableLoc = SheetBorderTop
    
    GetRowToSplitAfter = 0
    
    Dim i As Integer
    For i = rowStart To swTableAnn.rowCount - 1
        
        swTableAnn.SetRowHeight i, 4.30772602443226E-03, swTableRowColSizeChangeBehavior_e.swTableRowColChange_TableSizeCanChange
        TableLoc = TableLoc - swTableAnn.GetRowHeight(i)
        
        If TableLoc < ViewMaxLoc Then
            
            GetRowToSplitAfter = i - 1
            Exit For
            
        End If
    
    Next i
    
End Function

Private Function GetColumnWidth(swTable As SldWorks.TableAnnotation) As Double
    
    Dim i As Integer
    For i = 0 To swTable.ColumnCount - 1
                
        GetColumnWidth = GetColumnWidth + swTable.GetColumnWidth(i)
        
    Next i

End Function

