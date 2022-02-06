select
	count(to_char(rental_date, 'yyyy-mm-dd'))	
FROM
	rental
group by to_char(rental_date, 'yyyy-mm-dd'); 
	

select 
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'yyyy-mm'),
	to_char(rental_date, 'yyyy-mm-dd')
from
	rental;
	



select 
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'yyyy-mm'),
	to_char(rental_date, 'yyyy-mm-dd'),
	count(to_char(rental_date, 'yyyy-mm-dd'))
from
	rental
group by to_char(rental_date, 'yyyy-mm-dd');



-- ����
select 
	to_char(rental_date, 'yyyy') y,
	to_char(rental_date, 'mm') m,
	to_char(rental_date, 'dd') d,
	count(rental_id)
from
	rental
group by
rollup 
(
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd')
);


-- ��� ������� ���غ���
-- �����ϱ�
-- union all
-- group by
-- grouping sets
-- rollup 
-- cube


-- grouping sets
select 
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd'),
	count(rental_id)
from
	rental
group by
grouping sets
(
	(to_char(rental_date, 'yyyy'),to_char(rental_date, 'mm'),to_char(rental_date, 'dd')),
	(to_char(rental_date, 'yyyy'), to_char(rental_date, 'mm')),
 -- (to_char(rental_date, 'dd')),
	()
);


select 
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd'),
	count(rental_id)
from
	rental
group by
cube 
(
	to_char(rental_date, 'yyyy')
);




select 
	(rental_id)
from
	rental;

select 
	rental_id
from
	rental;


-- �⺰ �׷���̷� �̱�
select 
	to_char(rental_date, 'yyyy'),
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'yyyy');


-- ���� �׷���̷� �̱�
select 
	to_char(rental_date, 'yyyymm'),
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'yyyymm')
order by
	to_char(rental_date, 'yyyymm');


-- �Ϻ� �׷���̷� �̱�
select 
	to_char(rental_date, 'yyyymmdd'),
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'yyyymmdd')
order by
	to_char(rental_date, 'yyyymmdd');


-- ����
select 
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd'),
	count(rental_id)
from
	rental
group by
rollup
(
	to_char(rental_date, 'yyyy'),
	to_char(rental_date, 'mm'),
	to_char(rental_date, 'dd')
);
	

