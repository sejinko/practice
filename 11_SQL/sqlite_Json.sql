-- Json 파일
SELECT 
	fulltext
FROM 
	film
limit
	1;
	





WITH Split_Names ( userid, regdatetime, xmlname)
AS
(
    SELECT userid, regdatetime,  
    CONVERT(XML,'<Names><name>'  
    + REPLACE(EntityInfo,',', '</name><name>') + '</name></Names>') 
 AS xmlname
      FROM #entityinfo
)

 SELECT pid, regdatetime,   
 left(xmlname.value('/Names[1]/name[1]','varchar(100)'),9) AS index1,    
 left(xmlname.value('/Names[1]/name[2]','varchar(100)'),6) AS index2,
 left(xmlname.value('/Names[1]/name[3]','varchar(100)'),6) AS index3   -- index 뒤에 길이 파라미터는 확인하면서 바꿀 것
 into #entityinfo2
 FROM Split_Names