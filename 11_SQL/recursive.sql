create table tb_emp_recursive_test (
	employee_id serial primary key,
	full_name  varchar not null,
	manager_id int
	);
	
INSERT INTO TB_EMP_RECURSIVE_TEST (
EMPLOYEE_ID, FULL_NAME, MANAGER_ID)
VALUES
(1 , '이경오', NULL)
, (2 , '김한이', 1)
, (3 , '김승범', 1)
, (4 , '하선주', 1)
, (5 , '송백선', 1)
, (6 , '이슬이', 2)
, (7 , '홍발순', 2)
, (8 , '김미순', 2)
, (9 , '김선태', 2)
, (10, '이선형', 3)
, (11, '김선미', 3)
, (12, '김선훈', 3)
, (13, '이왕준', 3)
, (14, '김사원', 4)
, (15, '이시원', 4)
, (16, '최선영', 7)
, (17, '박태후', 7)
, (18, '최민준', 8)
, (19, '정택헌', 8)
, (20, '노가람', 8)
;

COMMIT;

select * from TB_EMP_RECURSIVE_TEST;


with recursive tmp1 as (
select 
	employee_id,
	manager_id,
	full_name,
	0 lvl
from
	tb_emp_recursive_test
where manager_id is null 
union 
select 
	e.employee_id,
	e.manager_id,
	e.full_name,
	s.lvl + 1
from
	tb_emp_recursive_test e,
	tmp1 s
where s.employee_id = e.manager_id
)

select 
	employee_id,
	manager_id,
	lpad(' ', 4 * (lvl)) || full_name as full_name
from
	tmp1;


with recursive tmp1 as (
select 
	employee_id,
	manager_id,
	full_name,
	0 lvl
from
	tb_emp_recursive_test 
where employee_id = 2
union 
select
	e.employee_id,
	e.manager_id,
	e.full_name,
	s.lvl + 1
from
	tb_emp_recursive_test e,
	tmp1 s
where s.employee_id = e.manager_id
)
select
	employee_id,
	manager_id
	lpad(' ', 4*(lvl) || full_name as full_name
from
	tmp1;
	