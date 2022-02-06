-- Json ����
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
 left(xmlname.value('/Names[1]/name[3]','varchar(100)'),6) AS index3   -- index �ڿ� ���� �Ķ���ʹ� Ȯ���ϸ鼭 �ٲ� ��
 into #entityinfo2
 FROM Split_Names