CREATE TABLE TB_MEMBER_NULLIF_TEST (
ID SERIAL PRIMARY KEY
, first_name VARCHAR (50) NOT NULL
, last_name VARCHAR (50) NOT NULL
, gender SMALLINT NOT NULL -- 1: male, 2 female
);
INSERT INTO TB_MEMBER_NULLIF_TEST (
FIRST_NAME
, LAST_NAME
, GENDER
)
VALUES
('John', 'Doe', 1)
, ('David', 'Dave', 1)
, ('Bush', 'Lily', 2)
;
COMMIT;


select * from tb_member_nullif_test;

select 
	(sum(case when gender = 1 then 1 else 0 end) / sum(case when gender = 2 then 1 else 0 end) ) * 100
	as "male/female ratio"
from
	tb_member_nullif_test;
	

UPDATE TB_MEMBER_NULLIF_TEST
SET GENDER = 1
WHERE GENDER = 2;
COMMIT;


select 
	(sum(case when gender = 1 then 1 else 0 end) / sum(case when gender = 2 then 1 else 0 end) ) * 100 as "male/female ratio"
from
	tb_member_nullif_test;
	

select 
	(sum(case when gender=1 then 1 else 0 end) / nullif(sum(case when gender=2 then 1 else 0 end),0))*100 as
	"male/female ratio"
from
	tb_member_nullif_test;