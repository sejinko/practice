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



-- 정답
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


-- 모든 방법으로 다해보기
-- 전부하기
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


-- 년별 그룹바이로 뽑기
select 
	to_char(rental_date, 'yyyy'),
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'yyyy');


-- 월별 그룹바이로 뽑기
select 
	to_char(rental_date, 'yyyymm'),
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'yyyymm')
order by
	to_char(rental_date, 'yyyymm');


-- 일별 그룹바이로 뽑기
select 
	to_char(rental_date, 'yyyymmdd'),
	count(rental_id)
from
	rental
group by
	to_char(rental_date, 'yyyymmdd')
order by
	to_char(rental_date, 'yyyymmdd');


-- 복귀
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
	

